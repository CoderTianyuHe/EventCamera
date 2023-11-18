img = np.full((800, 1280, 3), 255, dtype = "uint8")
        pos_events = eventData[eventData[:, 2] == 1, 3:]
        neg_events = eventData[eventData[:, 2] == 0, 3:]
        img[pos_events[:, 0], pos_events[:, 1]] = np.array([255, 0, 0], dtype = "uint8")
        img[neg_events[:, 0], neg_events[:, 1]] = np.array([0, 0, 255], dtype = "uint8")
        cv2.imshow('imshow', img)
        cv2.waitKey(32)
        cv2.destroyAllWindows()