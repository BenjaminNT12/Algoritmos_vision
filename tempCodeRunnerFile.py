 w, h = cv.boundingRect(contour)
            # aspect_ratio = float(w)/h

            # # Si el contorno es irregular, dibuja un contorno del mismo color que el fondo sobre él
            # if aspect_ratio < 0.9 or aspect_ratio > 1.1:
            #     cv.drawContou