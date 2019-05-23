my_string = str(input("Введите произвольную строку: "))
position_list = []
sub_string_list = []

# расчёт позиционных аргументов для создания множества подстрок
def position(my_string):
    for i in range(len(my_string) + 1):
        for j in range(i + 1, len(my_string) + 1):
            position_list.append((i, j))

position(my_string)

#преобразование заданной строки в множество подстрок
for key, value in position_list:
    sub_string_list.append(my_string[key : value])


for k in range(len(sub_string_list) + 1):
    print(sub_string_list.count(sub_string_list[k]))
    if sub_string_list.count(sub_string_list[k]) > 1:
        del sub_string_list[k]

print()
print("Список позиционных аргументов заданной подстроки :")
print(position_list)
print()
print("Подстроки :")
print(sub_string_list)
print()
print("В  строке " + my_string + " содержится " + str(len(sub_string_list)) + " подстрок")
print("И функции хэширования в этом ДЗ нет :)")
#print(sub_string_list.count(sub_string_list[k]))
