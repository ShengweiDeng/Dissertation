from predict import *
from tkinter import *

w2dic, model = loadmodel()

    # predict(w2dic, model, text)

window = Tk()
window.title("emotion analysis")
window.geometry("600x350")

entry_value = IntVar()  # IntVar is a class of tkinter that can manage radio buttons
entry_value2 = IntVar()  # IntVar is a class of tkinter that can manage radio buttons

def submit():
    text = message_entry.get('0.0', 'end')
    message_entry2.delete('1.0', 'end')
    res = predict(w2dic, model, text)
    message_entry2.insert('insert', res)
##text input 1
message_entry = Text(window,show=None,selectbackground="red",relief = GROOVE)
message_entry.place(x=40,y=100,width=260, height=150)
#show

entry = Label(window, text="please enter text：")
entry.place(x=40,y=70,width=100, height=30)
##text display input 2
message_entry2 = Text(window,show=None,selectbackground="red",relief = GROOVE)
message_entry2.place(x=300,y=100,width=240, height=150)

#translate button
button = Button(window, text="analyze",command = submit)
button.place(x=300,y=70,width=60, height=30) # Displaying the button


window.mainloop()  # message loop