from string import ascii_uppercase
from tensorflow.keras.models import load_model
import cv2
import tkinter as tk
import operator
from PIL import Image, ImageTk
from fast_autocomplete import AutoComplete

class Application:
    def __init__(self):
        self.directory = 'v4_epoch-10'
        self.vs = cv2.VideoCapture(0)
        self.current_image = None

        self.loaded_model = load_model(self.directory + "/model.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0

        print(self.ct)
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("India Sign Language GUI")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("667x660")
        self.panel = tk.Label(self.root)
        self.panel.place(x=10, y=38)
        # width = 640, height = 640
        self.T = tk.Label(self.root, fg='medium blue')
        self.T.place(x=25, y=6)
        self.T.config(text="Gesture Detection and Recognition of Indian Sign Language", font=("Times New Roman", 17, "bold"))

        # self.info3 = tk.Label(self.root)
        # self.info3.place(x=190, y=40)
        # self.info3.config(text="BY - Divyanshu Agarwal, Raghav Gupta and Antriksh Tiwari", font=("courier", 10, "normal"))

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=110, y=530)
        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=530)
        self.T1.config(text="Alphabet :", font=("Courier", 10, "bold"))

        self.panel4 = tk.Label(self.root)  # Word
        self.panel4.place(x=60, y=555)
        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=555)
        self.T2.config(text="Word :", font=("Courier", 10, "bold"))

        self.panel5 = tk.Label(self.root)  # Sentence
        self.panel5.place(x=110, y=580)
        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=580)
        self.T3.config(text="Sentence :", font=("Courier", 10, "bold"))

        self.panel6 = tk.Label(self.root)  # Prediction
        self.panel6.place(x=110, y=605)
        self.T4 = tk.Label(self.root)
        self.T4.place(x=10, y=615)
        self.T4.config(text="Prediction :", font=("Courier", 10, "bold"))

        # self.T4 = tk.Label(self.root)
        # self.T4.place(x=10, y=675)
        # self.T4.config(text="Suggestions", fg="red", font=("Courier", 10, "bold"))

        self.btcall = tk.Button(self.root, command=self.about_us, bg="snow3", fg="black")
        self.btcall.config(text="About Us", font=("Helvetica", 10, "italic"))
        self.btcall.place(x=583, y=530)

        self.btcall2 = tk.Button(self.root, command=self.help, bg="snow3", fg="black")
        self.btcall2.config(text="Help", font=("Helvetica", 10, "italic"))
        self.btcall2.place(x=612, y=563)

        self.bt1 = tk.Button(self.root, command=self.nlp)
        self.bt1.config(text="")
        self.bt1.place(x = 120, y = 610)

        self.bt2 = tk.Button(self.root, command=self.nlp1)
        self.bt2.config(text="")
        self.bt2.place(x=180, y=610)
        # self.bt2.grid(padx=200, pady=610)

        self.bt3 = tk.Button(self.root, command=self.nlp2)
        self.bt3.config(text="")
        self.bt1.place(x=230, y=610)
        # self.bt3.grid(padx=275, pady=610)

        self.bt4 = tk.Button(self.root, command=self.clear,width=10)
        self.bt4.config(text="Clear")
        self.bt4.place(x=290, y=480)

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.words = {'hello': {}, 'there': {}, 'good': {}, 'how': {},'are':{},"you":{},"good":{},'wonderful':{},
                      'very':{},'hot':{},"cool":{},"I'm":{},"was":{},"is":{},"your":{},"day":{}}
        self.video_loop()

    def clear(self):
        self.word= ""
        self.str = ""
        self.current_symbol = "Empty"

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()

    def destructor2(self):
        print("Closing Application...")
        self.root2.destroy()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            cv2.rectangle(cv2image, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            self.predict(res)
            # self.current_image2 = Image.fromarray(res)
            # imgtk = ImageTk.PhotoImage(image=self.current_image2)
            # self.panel2.imgtk = imgtk
            # self.panel2.config(image=imgtk)

            self.panel3.config(text=self.current_symbol,font=("Courier",10))
            self.panel4.config(text=self.word,font=("Courier",10))
            self.panel5.config(text=self.str,font=("Courier",10))

            predicts = self.word
            autocomplete = AutoComplete(words=self.words)
            self.a = autocomplete.search(word=predicts, max_cost=2, size=2)
            print("Initial {0}: ,Suggest : {1}".format(predicts,self.a))
            if (len(self.a) > 0 ):
                self.bt1.config(text=self.a[0][0], font=("Courier", 10))
            else:
                self.bt1.config(text="None")

            if (len(self.a) > 1 ):
                self.bt2.config(text=self.a[1][0], font=("Courier", 10))
            else:
                self.bt2.config(text="None")

            if (len(self.a) > 2 ):
                self.bt3.config(text=self.a[2][0], font=("Courier", 10))
            else:
                self.bt3.config(text="None")


        self.root.after(60, self.video_loop)

    def nlp(self):
        suggest=self.a[0][0]
        if(len(suggest) > 0):
            self.word=""
            self.str+=" "
            self.str+=suggest

    def nlp1(self):
        suggest=self.a[1][0]
        if(len(suggest) > 0):
            self.word=""
            self.str+=" "
            self.str+=suggest

    def nlp2(self):
        suggest=self.a[2][0]
        if(len(suggest) > 0):
            self.word=""
            self.str+=" "
            self.str+=suggest

    def predict(self,test_image):
        test_image = cv2.resize(test_image, (128,128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))

        prediction={}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        # print(prediction)

        #OUTPUT OF MODEL
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        print(prediction)
        self.current_symbol = prediction[0][0]

        #GUI CONTROL
        if(self.current_symbol == 'blank'):
            print("yes")
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1
        print(self.ct)
        # print("="*100)
        if(self.ct[self.current_symbol] > 13):
            print("Inside >15")
            print("="*100)

            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                print("here")
                print(self.ct)
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if(len(self.str) > 16):
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def about_us(self):

        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("425x350")

        self.tx = tk.Label(self.root1)
        self.tx.place(x=150, y=10)
        self.tx.config(text="Contributors", fg="firebrick1", font=("David", 15, "underline"))


        img = Image.open("C:/Users/Raghu/Desktop/Gestures/pics/raghav.jpeg")
        img = img.resize((110, 180), Image.ANTIALIAS)
        self.photo1 = ImageTk.PhotoImage(img)
        self.w1 = tk.Label(self.root1, image=self.photo1)
        self.w1.place(x=10, y=40)
        self.tx2 = tk.Label(self.root1,bg='LightBlue4',fg="snow")
        self.tx2.place(x=12, y=227)
        self.tx2.config(text=" Raghav Gupta  ", font=("Courier", 9, "bold"))

        img2 = Image.open("C:/Users/Raghu/Desktop/Gestures/pics/divyanshu.jpeg")
        img2=img2.resize((121,180), Image.ANTIALIAS)
        self.photo2 = ImageTk.PhotoImage(img2)
        self.w2 = tk.Label(self.root1, image=self.photo2)
        self.w2.place(x=150, y=40)
        self.tx3 = tk.Label(self.root1,bg='LightBlue4',fg="snow")
        self.tx3.place(x=150, y=227)
        self.tx3.config(text="Divyanshu Agarwal", font=("Courier", 9, "bold"))

        img3 = Image.open("C:/Users/Raghu/Desktop/Gestures/pics/antriksh.jpeg")
        img3 = img3.resize((110,180), Image.ANTIALIAS)
        self.photo3 = ImageTk.PhotoImage(img3)
        self.w3 = tk.Label(self.root1, image=self.photo3)
        self.w3.place(x=301, y=40)
        self.tx4 = tk.Label(self.root1,bg='LightBlue4',fg="snow")
        self.tx4.place(x=303, y=227)
        self.tx4.config(text="Antriksh Tiwari", font=("Courier", 9, "bold"))

        self.tx5 = tk.Label(self.root1)
        self.tx5.place(x=120, y=270)
        self.tx5.config(text="Under the guidance of ", fg="firebrick1", font=("David", 15, "underline"))

        self.tx6 = tk.Label(self.root1)
        self.tx6.place(x=15, y=300)
        self.tx6.config(text="Professor Dharmesh Rathod", fg="medium blue", font=("Times New Roman", 24,["italic","bold"]))

    def help(self):

        self.root2 = tk.Toplevel(self.root)
        self.root2.title("About")
        self.root2.protocol('WM_DELETE_WINDOW', self.destructor2)
        self.root2.geometry("650x300")

        img = Image.open("C:/Users/Raghu/Desktop/Gestures/pics/isl_2.jpg")
        self.photo1 = ImageTk.PhotoImage(img)
        self.w1 = tk.Label(self.root2, image=self.photo1)
        self.w1.place(x=10, y=10)
        # self.tx2 = tk.Label(self.root1, bg='LightBlue4', fg="snow")
        # self.tx2.place(x=12, y=227)
        # self.tx2.config(text=" Raghav Gupta  ", font=("Courier", 9, "bold"))


pba = Application()
pba.root.mainloop()