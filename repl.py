from poet import Poet


EPOCHS = 10




my_poet = Poet()
gen, total_g, total_d = my_poet.generate(False)


input_name = str(input('Input the name of the person you want a poem generated about:\n'))
input_characteristic = str(input('Input one characteristic about this person:\n'))


while True:
    try:
        poem = my_poet.create_poem(input_name, input_characteristic)
        print(poem)


        input_name = str(input('Input the name of the person you want a poem generated about:\n'))
        input_characteristic = str(input('Input one characteristic about this person:\n'))
        gen, total_g, total_d = my_poet.generate(True)
    except KeyboardInterrupt:
        break
