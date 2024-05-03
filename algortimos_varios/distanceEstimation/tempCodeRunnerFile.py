 int(avg_radius) < 10:
            print("se activo", avg_radius)
            kernel = np.ones((5,5),np.uint8)
            mask_long_distance = cv.inRange(frame_to_process, lower_color2, upper_color2)
            # Dilatar la imagen
            dilation = cv.dilate(mask_long_distance, kernel, iterations = 1)
            mask_long_distance = cv.medianBlur(dilation, 5)
            cv.imshow('mask2', mask_long_distance)
            frame_to_detect_binary = cv.add(frame_to_detect_binary, mask_long_distance)
        else:
            print("se desactivo")