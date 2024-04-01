class Det:
    def __init__(self, masks):
        #self.xyxy = ann.boxes.xyxy.numpy()
        #self.class_id = ann.boxes.cls.numpy().astype(int)
        self.mask = masks
        self.xyxy = np.array(list(map(lambda mask: self.mask_to_bbox(mask), masks)))
        self.area = np.array(list(map(lambda xyxy: (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1]), self.xyxy)))
        self.class_id = np.array(list(map(lambda mask: 0, self.mask)))
        
    def mask_to_bbox(self, mask):
        positive_cols = np.where(np.any(mask, axis=0))[0]
        positive_rows = np.where(np.any(mask, axis=1))[0]
        min_x, max_x, min_y, max_y = positive_cols.min(), positive_cols.max(), positive_rows.min(), positive_rows.max()
        return min_x, min_y, max_x, max_y
    
    def __len__(self):
        return len(self.xyxy)