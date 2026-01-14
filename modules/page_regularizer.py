"""
Page Regularizer Module
- Page Orientation Correction
- Page Size Normalization

The meaning of the output parameters is as follows: 
- input_path: Represents the path of the input image. 
- class_ids: Represents the predicted class ID, with four categories: 0°, 90°, 180°, and 270°.
- scores: Represents the confidence level of the prediction result.
- label_names: Represents the category names of the prediction results.

Usage:
    from modules.page_regularizer import PageRegularizer
    
    # Initialize
    regularizer = PageRegularizer()
    
    # Regularize image
    result = regularizer.regularize(input_path, output_path)
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any

try:
    from paddleocr import DocImgOrientationClassification
except ImportError:
    print("Warning: paddleocr not installed. Install with: pip install paddleocr")
    DocImgOrientationClassification = None


class PageRegularizer:
    """Page regularizer for orientation correction and normalization"""
    
    def __init__(self, model_name: str = "PP-LCNet_x1_0_doc_ori", verbose: bool = True):
        """Initialize the page regularizer
        
        Args:
            model_name: Model name for orientation classification
            verbose: Print initialization messages
        """
        if DocImgOrientationClassification is None:
            raise ImportError("paddleocr is required. Install with: pip install paddleocr")
        
        self.model = DocImgOrientationClassification(model_name=model_name)
        self.verbose = verbose
        if verbose:
            print(f"PageRegularizer initialized with model: {model_name}")
    
    def regularize(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Regularize a single image (orientation correction)
        
        Args:
            image_path: Path to input image
            output_path: Path to save corrected image (optional, if None only returns prediction)
            
        Returns:
            Dictionary containing:
                - input_path: Input image path
                - output_path: Output image path (if saved)
                - class_id: Predicted orientation class (0=0°, 1=90°, 2=180°, 3=270°)
                - score: Confidence score
                - label_name: Orientation label (e.g., "0", "90", "180", "270")
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Predict orientation
        output = self.model.predict(image_path, batch_size=1)
        
        result = {
            "input_path": image_path,
            "output_path": output_path,
            "class_id": None,
            "score": None,
            "label_name": None
        }
        
        for res in output:
            # Extract results - PaddleX returns TopkResult which is dict-like
            if isinstance(res, dict) or hasattr(res, 'keys'):
                result["class_id"] = res.get('class_ids', [None])[0] if 'class_ids' in res else res.get('class_id')
                result["score"] = res.get('scores', [None])[0] if 'scores' in res else res.get('score')
                result["label_name"] = res.get('label_names', [None])[0] if 'label_names' in res else res.get('label_name')
            else:
                # Fallback to attributes (if API changes)
                if hasattr(res, 'class_ids') and len(res.class_ids) > 0:
                    result["class_id"] = int(res.class_ids[0])
                if hasattr(res, 'scores') and len(res.scores) > 0:
                    result["score"] = float(res.scores[0])
                if hasattr(res, 'label_names') and len(res.label_names) > 0:
                    result["label_name"] = res.label_names[0]
            
            # Save corrected image if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                res.save_to_img(output_path)
                if self.verbose:
                    print(f"Saved corrected image to: {output_path}")
        
        return result
    
    def batch_regularize(self, image_paths: list, output_dir: Optional[str] = None) -> list:
        """Regularize multiple images
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory to save corrected images (optional)
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for img_path in image_paths:
            if output_dir:
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(output_dir, f"{name}_regularized{ext}")
            else:
                output_path = None
            
            result = self.regularize(img_path, output_path)
            results.append(result)
        
        return results


def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description="Page Regularizer - Orientation Correction")
    parser.add_argument("input", type=str, help="Input image path or directory")
    parser.add_argument("--output", type=str, help="Output image path or directory")
    parser.add_argument("--batch", action="store_true", help="Process directory of images")
    
    args = parser.parse_args()
    
    # Initialize regularizer
    reg = PageRegularizer()
    
    if args.batch:
        # Batch processing
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            sys.exit(1)
        
        # Get all images
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for fname in os.listdir(args.input):
            if any(fname.lower().endswith(ext) for ext in image_exts):
                image_paths.append(os.path.join(args.input, fname))
        
        if not image_paths:
            print(f"No images found in {args.input}")
            sys.exit(1)
        
        print(f"Processing {len(image_paths)} images...")
        results = reg.batch_regularize(image_paths, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PROCESSING RESULTS")
        print("="*60)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {os.path.basename(result['input_path'])}")
            print(f"   Orientation: {result['label_name']}° (confidence: {result['score']:.4f})")
            if result['output_path']:
                print(f"   Saved to: {result['output_path']}")
    
    else:
        # Single image processing
        if not os.path.exists(args.input):
            print(f"Error: {args.input} not found")
            sys.exit(1)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            name, ext = os.path.splitext(args.input)
            output_path = f"{name}_regularized{ext}"
        
        # Process image
        result = reg.regularize(args.input, output_path)
        
        # Print results
        print("\n" + "="*60)
        print("REGULARIZATION RESULTS")
        print("="*60)
        print(f"Input:       {result['input_path']}")
        print(f"Output:      {result['output_path']}")
        print(f"Orientation: {result['label_name']}°")
        print(f"Class ID:    {result['class_id']}")
        print(f"Confidence:  {result['score']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()