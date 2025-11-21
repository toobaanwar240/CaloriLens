# segmentation/segmentation.py

class SegmentationModel:
    """
    Wrapper class to switch between YOLOv8 segmentation or SAM.
    """

    def __init__(self, model_type="yolo", model_path=None, sam_checkpoint=None):
        """
        model_type: "yolo" or "sam"
        model_path: path to YOLOv8 segmentation model (.pt)
        sam_checkpoint: path to SAM model checkpoint (.pth)
        """
        self.model_type = model_type.lower()

        if self.model_type == "yolo":
            from .yolo_seg import FoodSegmentationYOLO
            self.model = FoodSegmentationYOLO(model_path=model_path)

        elif self.model_type == "sam":
            from .sam_seg import FoodSegmentationSAM
            self.model = FoodSegmentationSAM(checkpoint_path=sam_checkpoint)

        else:
            raise ValueError("model_type must be 'yolo' or 'sam'")

    def segment(self, image):
        """
        Returns:
            bboxes: [x1, y1, x2, y2, confidence]
            masks: list of binary segmentation masks
        """
        return self.model.segment(image)
