# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

import Levenshtein
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque
from tqdm import tqdm


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == 'td':
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % \
                     (self.tag, self.colspan, self.rowspan, self.content)
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value
        """
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1
        """
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.
        if node1.tag == 'td':
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.


class TEDS(object):
    ''' Tree Edit Distance basead Similarity
    '''
    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), 'n_jobs must be an integer greather than 1'
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        ''' Tokenizes table cells
        '''
        self.__tokens__.append('<%s>' % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != 'unk':
            self.__tokens__.append('</%s>' % node.tag)
        if node.tag != 'td' and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        ''' Converts HTML tree to the format required by apted
        '''
        global __tokens__
        if node.tag == 'td':
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(node.tag,
                                 int(node.attrib.get('colspan', '1')),
                                 int(node.attrib.get('rowspan', '1')),
                                 cell, *deque())
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != 'td':
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        ''' Computes TEDS score between the prediction and the ground truth of a
            given sample
        '''
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath('body/table') and true.xpath('body/table'):
            pred = pred.xpath('body/table')[0]
            true = true.xpath('body/table')[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0

    def batch_evaluate(self, pred_json, true_json):
        ''' Computes TEDS score between the prediction and the ground truth of
            a batch of samples
            @params pred_json: {'FILENAME': 'HTML CODE', ...}
            @params true_json: {'FILENAME': {'html': 'HTML CODE'}, ...}
            @output: {'FILENAME': 'TEDS SCORE', ...}
        '''
        samples = true_json.keys()
        scores = [self.evaluate(pred_json.get(filename, ''), true_json[filename]['html']) for filename in tqdm(samples)]
        scores = dict(zip(samples, scores))
        return scores


# =============================================================================
# HELPER FUNCTIONS FOR REWARD INTEGRATION
# =============================================================================

def wrap_table_html(table_html: str) -> str:
    """
    Wrap a table HTML string in <html><body> tags for TEDS evaluation.
    
    Args:
        table_html: HTML string starting with <table> tag
        
    Returns:
        Wrapped HTML string
    """
    table_html = table_html.strip()
    if not table_html.startswith('<html>'):
        table_html = f'<html><body>{table_html}</body></html>'
    return table_html


def compute_teds(pred_table: str, true_table: str, structure_only: bool = False) -> float:
    """
    Compute TEDS score between predicted and ground truth table.
    
    Args:
        pred_table: Predicted table HTML (can be raw <table>...</table> or wrapped)
        true_table: Ground truth table HTML
        structure_only: If True, compute TEDS-S (structure only), else TEDS (with content)
        
    Returns:
        TEDS score between 0 and 1 (1 = perfect match)
    """
    try:
        pred_wrapped = wrap_table_html(pred_table)
        true_wrapped = wrap_table_html(true_table)
        
        teds_evaluator = TEDS(structure_only=structure_only)
        score = teds_evaluator.evaluate(pred_wrapped, true_wrapped)
        return score
    except Exception as e:
        # Return 0 if evaluation fails (malformed HTML, etc.)
        return 0.0


def compute_teds_s(pred_table: str, true_table: str) -> float:
    """
    Compute TEDS-S (structure only) score.
    
    Args:
        pred_table: Predicted table HTML
        true_table: Ground truth table HTML
        
    Returns:
        TEDS-S score between 0 and 1
    """
    return compute_teds(pred_table, true_table, structure_only=True)


if __name__ == '__main__':
    # Test the TEDS implementation
    print("Testing TEDS implementation...")
    
    # Sample tables for testing
    table1 = """
    <table>
        <tr>
            <td>Container No.</td>
            <td>Weight</td>
            <td>Measurement</td>
        </tr>
        <tr>
            <td>MSKU1234567</td>
            <td>15000 KGS</td>
            <td>30 CBM</td>
        </tr>
    </table>
    """
    
    table2_identical = """
    <table>
        <tr>
            <td>Container No.</td>
            <td>Weight</td>
            <td>Measurement</td>
        </tr>
        <tr>
            <td>MSKU1234567</td>
            <td>15000 KGS</td>
            <td>30 CBM</td>
        </tr>
    </table>
    """
    
    table3_different_content = """
    <table>
        <tr>
            <td>Container No.</td>
            <td>Weight</td>
            <td>Measurement</td>
        </tr>
        <tr>
            <td>TCLU7654321</td>
            <td>18000 KGS</td>
            <td>35 CBM</td>
        </tr>
    </table>
    """
    
    table4_different_structure = """
    <table>
        <tr>
            <td>Container</td>
            <td>Weight</td>
        </tr>
        <tr>
            <td>MSKU1234567</td>
            <td>15000</td>
        </tr>
        <tr>
            <td>Total</td>
            <td>15000</td>
        </tr>
    </table>
    """
    
    table5_empty = ""
    
    print("\n" + "=" * 60)
    print("Test 1: Identical tables")
    teds_score = compute_teds(table1, table2_identical, structure_only=False)
    teds_s_score = compute_teds(table1, table2_identical, structure_only=True)
    print(f"  TEDS:   {teds_score:.4f} (expected: 1.0)")
    print(f"  TEDS-S: {teds_s_score:.4f} (expected: 1.0)")
    
    print("\n" + "=" * 60)
    print("Test 2: Same structure, different content")
    teds_score = compute_teds(table1, table3_different_content, structure_only=False)
    teds_s_score = compute_teds(table1, table3_different_content, structure_only=True)
    print(f"  TEDS:   {teds_score:.4f} (expected: < 1.0, content differs)")
    print(f"  TEDS-S: {teds_s_score:.4f} (expected: 1.0, structure same)")
    
    print("\n" + "=" * 60)
    print("Test 3: Different structure")
    teds_score = compute_teds(table1, table4_different_structure, structure_only=False)
    teds_s_score = compute_teds(table1, table4_different_structure, structure_only=True)
    print(f"  TEDS:   {teds_score:.4f} (expected: < 1.0)")
    print(f"  TEDS-S: {teds_s_score:.4f} (expected: < 1.0)")
    
    print("\n" + "=" * 60)
    print("Test 4: Empty table (edge case)")
    teds_score = compute_teds(table1, table5_empty, structure_only=False)
    teds_s_score = compute_teds(table1, table5_empty, structure_only=True)
    print(f"  TEDS:   {teds_score:.4f} (expected: 0.0)")
    print(f"  TEDS-S: {teds_s_score:.4f} (expected: 0.0)")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
