from poet import Poet

EPOCHS = 10

def run_poetry_generator():
    """Runs the poetry generator.
    Args: n/a
    Returns: n/a
    Terminates: press control/command c in terminal
    """
    my_poet = Poet()
    # Train model the first time, otherwise load the model
    my_poet.generate(False, EPOCHS)

    print("To terminate, press control c\n")
    input_name = str(input('Input the name of the person you want a poem generated about:\n'))
    input_characteristic = str(input('Input one characteristic about this person:\n'))

    while True:
        try:
            poem = my_poet.create_poem(input_name, input_characteristic)
            print(poem)
            print('\n')

            print("To terminate, press control c\n")
            input_name = str(input('Input the name of the person you want a poem generated about:\n'))
            input_characteristic = str(input('Input one characteristic about this person:\n'))
            my_poet.generate(True, EPOCHS)
        
        except KeyboardInterrupt:
            print("Thank you for using Poetic Prodigy!")
            break


# Call method to run program
run_poetry_generator()