import os
# os.getcwd()
letter="A"
collection = "C:/Users/Raghu/Desktop/Gestures/Confusion Matrix/test/"+letter
a = 633
for i, filename in enumerate(os.listdir(collection)):
    i=i+a
    try:
        os.rename(collection+"/" + filename, collection+"/" + str(i) + ".jpg")
    except FileExistsError:
        os.rename(collection+"/" + filename,
                  collection+"/" + str(i) + "(1)" + ".jpg")
