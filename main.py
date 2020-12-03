import face_datasets 
import training
import face_recognition 

while(1):
    print("1 : dataset")
    print("2 : training")
    print("3 : recognition")
    ##print("4 : check out")
    ##print("5 : update data to could")
    print("0 : Exit program")
    print("\n------------------------------------")
    choice = int(input("Please input function : "))

    if(choice is 1):
        face_datasets.face_dataset()
    elif (choice is 2):
        training.training()
    elif (choice is 3):
        print("Back to main program please button q")
        face_recognition.recognition()
    else:
        print("error")
