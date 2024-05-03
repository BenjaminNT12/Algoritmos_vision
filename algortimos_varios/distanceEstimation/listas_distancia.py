point3 = [17, 1]
point1 = [15, 16]
point2 = [2, 10]
point4 = [3, 3]

# pos = 2*[[0 for i in range(2)]]

matriz = []

matriz.append(point3)
# print(matriz)

# list = [1,2,3,4,153,92,53]
# pos = [point3, point1, point2, point4]


# max_x = vec_x[vec_x.index(max(vec_x))]
# max_y = vec_y[vec_x.index(max(vec_x))]

# print(max_x)
# print(max_y)


def order_points(vec_x, vec_y):
    print(vec_x)
    print(vec_y)
    res = []
    posiciones = []
    for i in range(len(vec_x)):
        res.append(vec_x[i] + vec_y[i])

    posiciones.append(res.index(max(res)))
    posiciones.append(res.index(min(res)))

    # print(posiciones)

    # res.pop(posiciones[0])
    # res.pop(posiciones[1])

    print(posiciones)

    temp = 0
    for i in range(len(vec_x)):
        if i not in posiciones:
            if vec_x[i] > temp:
                posiciones.append(i)
            else:
                posiciones.insert(2, i)
            temp = vec_x[i]
    
    points = [ [vec_x[posiciones[1]], vec_y[posiciones[1]]]
              ,[vec_x[posiciones[3]], vec_y[posiciones[3]]]
              ,[vec_x[posiciones[0]], vec_y[posiciones[0]]]
              ,[vec_x[posiciones[2]], vec_y[posiciones[2]]] ]
    # points = list(zip(vec_x, vec_y))
    print(points)

    print(posiciones)
    return points

    # print(res)
    # print(max(res))
    # print(res.index(max(res)))
    # print(res.index(max(res)))

    
    return posiciones



def main():
    # vec_x = [126, 568, 558, 124]
    # vec_y = [192, 184, 25, 28]
    # vec_x = [632, 115, 619, 130]
    # vec_y = [273, 271, 58, 54]
    vec_x = [0, 0, 406, 410]
    vec_y = [196, 185, 188, 25]
    # vec_x = [224, 949, 175, 212]
    # vec_y = [414, 339, 49, 85]
    # vec_x = [307, 850, 291, 828]
    # vec_y = [362, 331, 112, 86]
    # vec_x = [1, 2, 15, 14]
    # vec_y = [10,8, 14, 15]

    # vec_x = [386, 732, 382, 711]
    # vec_y = [269, 227, 128, 52]

    print(vec_x)
    print(vec_y)
    # vec_x.
    # print(vec_x)

    posicion = order_points(vec_x, vec_y)
    print(posicion)

if __name__ == "__main__":
    main()

# for i in range(len(vec_x)):




# for i in range(len(pos[:][:])):

#     print(i)





# print(point1)