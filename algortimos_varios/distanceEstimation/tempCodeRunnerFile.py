
                   "Sec: {:.2f}".format(time.time()-sec),
                   (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0), 1, 
                   cv.LINE_AA)   
        
        
        cv.putText(frame_to_process, 
                   "FPS: {:.2f}".format(1 / (end - start)),
                   (10, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0), 1, 
                   cv.LINE_AA)