or contour in contours:
#         # Calcular el área del contorno
#         area = cv.contourArea(contour)
#         # print("area: ", area)
#         # Descartar contornos pequeños
#         print("area: ", area)
#         if area > AREA_THRESHOLD:
#             print(" aceptada")
#             # Calcular el centroide del contorno
#             moment = cv.moments(contour)
#             centroid_x = int(moment["m10"] / moment["m00"])
#             centroid_y = int(moment["m01"] / moment["m00"])

#             if PTS_COMPLETE is False:
#                 coordenadas.append([centroid_x, centroid_y])
#                 new_cordinates = np.array(coordenadas)
#             else:
#                 actual_coordinates = np.array([centroid_x, centroid_y])
#                 position = dentro_de_area(
#                     new_cordinates, actual_coordinates, 50)
#                 np.put(new_cordinates, [len(actual_coordinates)*position,
#                        len(actual_coordinates)*position + 1], actual_coordinates)

#                 translacion, angle = calcular_pose(new_cordinates) # Se calcula la translacion y el angulo

#                 cv.circle(frame_to_track, (int(translacion[0]), int(
#                     translacion[1])), 5, (0, 0, 0), -1)

#                 draw_line(frame_to_track, new_cordinates[0][:],
#                           new_cordinates[1][:], thickness=3)
#                 draw_line(frame_to_track, new_cordinates[1][:],
#                           new_cordinates[3][:], thickness=3)
#                 draw_line(frame_to_track, new_cordinates[3][:],
#                           new_cordinates[2][:], thickness=3)
#                 draw_line(frame_to_track, new_cordinates[2][:],
#                           new_cordinates[0][:], thickness=3)
# ############################################################################################################
#                 plot_points(frame_to_track, objectPoints)
#                 coordenadas_float = np.array(new_cordinates, dtype=np.float32)

#                 if (len(new_cordinates) > NUMERO_DE_PUNTOS-1):
#                     _, rotacion3d, translacion3D = estimar_pose_3D(objectPoints,
#                                                                    coordenadas_float,
#                                                                    cameraMatrix,
#                                                                    distCoeffs) # Se calcula la rotacion y la translacion en 3D
#                     print("translacion: ", translacion3D)
#                     cv.putText(frame_to_track,
#                                "Rotacion X: " + str(int(math.degrees(rotacion3d[0]))),
#                                (20, 60), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
#                     cv.putText(frame_to_track,
#                                "Rotacion Y: " + str(int(math.degrees(rotacion3d[1]))),
#                                (20, 80), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
#                     cv.putText(frame_to_track, 
#                                "Rotacion Z: " + str(int(math.degrees(rotacion3d[2]))),
#                                (20, 100), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                    
#                     nose_end_point2D, jacobian = cv.projectPoints(
#                         np.array([(0.0, 0.0, 1000.0)]),
#                         rotacion3d,
#                         translacion3D,
#                         cameraMatrix,
#                         distCoeffs)

#                     point1 = (int(new_cordinates[0][0]),
#                               int(new_cordinates[0][1]))

#                     point2 = (int(nose_end_point2D[0][0][0]),
#                               int(nose_end_point2D[0][0][1]))
#                     print(nose_end_point2D)
#                     cv.line(frame_to_track, point1, point2, (0, 0, 0), 2)
# ############################################################################################################
#                 d1cm, d2cm, d3cm, d4cm = calculate_distance(new_cordinates)

#                 cv.putText(frame_to_track, "Angulo: " + str(int(angle)) + " Grados",
#                            (20, 20), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

#                 cv.putText(frame_to_track, "Distancias: "
#                            + str(int(d1cm)) + "cm ,"
#                            + str(int(d2cm)) + "cm ,"
#                            + str(int(d3cm)) + "cm ,"
#                            + str(int(d4cm)) + "cm ",
#                            (20, 40), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

#             cv.putText(frame_to_track, str(position), (centroid_x - 25, centroid_y - 25),
#                        cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
#             cv.circle(frame_to_track, (centroid_x, centroid_y),
#                       12, (0, 255, 0), -1)