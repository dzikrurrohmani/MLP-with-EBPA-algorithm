import tkinter
import numpy as np
from datetime import datetime
from tkinter import ttk
from tkinter import scrolledtext as tkst
from tkinter import messagebox as mb
from tkinter.filedialog import askopenfilename
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Declaration of Additional Variable
font9 = "-family {Showcard Gothic} -size 14 -weight bold " \
        "-slant roman -underline 0 -overstrike 0"

# Make a Tkinter Canvas
windows = tkinter.Tk()
windows.title("MLP with EBPA for Simulating Gait Phase Data Recognition")
windows.configure(background="#e2e6e2")
windows.geometry("+250+10")

# Declare Attribute
Title = tkinter.Label(windows, text="Multi Layer Perceptron with Error Back Propagation Algorithm", font=font9,
                      bg='#e2e6e2')
Title.pack(pady=5, expand=tkinter.YES, side=tkinter.TOP)
part_1 = tkinter.Frame(windows, background='#e2e6e2')
part_1.pack()
part_2 = tkinter.Frame(windows, background='#e2e6e2')
part_2.pack()

my_frame_1 = tkinter.LabelFrame(part_1, width=103, height=75, text='''Training Data''', relief='groove', foreground="black",
                                background='#e2e6e2')
my_frame_1.pack(padx=5, side=tkinter.LEFT)
my_frame_1.pack_propagate(0)
inputan = tkinter.StringVar(my_frame_1)
my_frame_1_combobox = ttk.Combobox(my_frame_1, width=8, textvariable=inputan)
my_frame_1_combobox['values'] = ('Topology A', 'Topology B')
my_frame_1_combobox.configure(justify=tkinter.CENTER)
my_frame_1_combobox.pack(pady=4)

my_frame_2 = tkinter.LabelFrame(part_1, width=400, height=75, text='''ANN Topology''', relief='groove',
                                foreground="black", background='#e2e6e2')
my_frame_2.pack(side=tkinter.LEFT)
my_frame_2.pack_propagate(0)
my_frame_2_bantuan1 = tkinter.Frame(my_frame_2, width=85, height=50, background='#e2e6e2')
my_frame_2_bantuan1.pack(side=tkinter.LEFT)
my_frame_2_bantuan2 = tkinter.Frame(my_frame_2, width=65, height=50, background='#e2e6e2')
my_frame_2_bantuan2.pack(side=tkinter.LEFT)
my_frame_2_bantuan3 = tkinter.Frame(my_frame_2, width=50, height=50, background='#e2e6e2')
my_frame_2_bantuan3.pack(side=tkinter.LEFT)
my_frame_2_bantuan4 = tkinter.Frame(my_frame_2, width=85, height=50, background='#e2e6e2')
my_frame_2_bantuan4.pack(side=tkinter.LEFT)
my_frame_2_bantuan5 = tkinter.Frame(my_frame_2, width=65, height=50, background='#e2e6e2')
my_frame_2_bantuan5.pack(side=tkinter.LEFT)
my_frame_2_bantuan6 = tkinter.Frame(my_frame_2, width=50, height=50, background='#e2e6e2')
my_frame_2_bantuan6.pack(side=tkinter.LEFT)
my_frame_2_bantuan7 = tkinter.Frame(my_frame_2_bantuan1, width=85, height=25, background='#e2e6e2')
my_frame_2_bantuan7.pack()
my_frame_2_bantuan7.pack_propagate(0)
my_frame_2_bantuan8 = tkinter.Frame(my_frame_2_bantuan1, width=85, height=25, background='#e2e6e2')
my_frame_2_bantuan8.pack()
my_frame_2_bantuan8.pack_propagate(0)
my_frame_2_bantuan9 = tkinter.Frame(my_frame_2_bantuan2, width=65, height=25, background='#e2e6e2')
my_frame_2_bantuan9.pack()
my_frame_2_bantuan9.pack_propagate(0)
my_frame_2_bantuan10 = tkinter.Frame(my_frame_2_bantuan2, width=65, height=25, background='#e2e6e2')
my_frame_2_bantuan10.pack()
my_frame_2_bantuan10.pack_propagate(0)
my_frame_2_bantuan11 = tkinter.Frame(my_frame_2_bantuan3, width=50, height=25, background='#e2e6e2')
my_frame_2_bantuan11.pack()
my_frame_2_bantuan11.pack_propagate(0)
my_frame_2_bantuan12 = tkinter.Frame(my_frame_2_bantuan3, width=50, height=25, background='#e2e6e2')
my_frame_2_bantuan12.pack()
my_frame_2_bantuan12.pack_propagate(0)
my_frame_2_bantuan13 = tkinter.Frame(my_frame_2_bantuan4, width=85, height=25, background='#e2e6e2')
my_frame_2_bantuan13.pack()
my_frame_2_bantuan13.pack_propagate(0)
my_frame_2_bantuan14 = tkinter.Frame(my_frame_2_bantuan4, width=85, height=25, background='#e2e6e2')
my_frame_2_bantuan14.pack()
my_frame_2_bantuan14.pack_propagate(0)
my_frame_2_bantuan15 = tkinter.Frame(my_frame_2_bantuan5, width=65, height=25, background='#e2e6e2')
my_frame_2_bantuan15.pack()
my_frame_2_bantuan15.pack_propagate(0)
my_frame_2_bantuan16 = tkinter.Frame(my_frame_2_bantuan5, width=65, height=25, background='#e2e6e2')
my_frame_2_bantuan16.pack()
my_frame_2_bantuan16.pack_propagate(0)
my_frame_2_bantuan17 = tkinter.Frame(my_frame_2_bantuan6, width=50, height=25, background='#e2e6e2')
my_frame_2_bantuan17.pack()
my_frame_2_bantuan17.pack_propagate(0)
my_frame_2_bantuan18 = tkinter.Frame(my_frame_2_bantuan6, width=50, height=25, background='#e2e6e2')
my_frame_2_bantuan18.pack()
my_frame_2_bantuan18.pack_propagate(0)
my_frame_2_text1 = tkinter.Label(my_frame_2_bantuan7, text='Input Layer', background='#e2e6e2')
my_frame_2_text1.pack(side=tkinter.LEFT)
my_frame_2_text2 = tkinter.Label(my_frame_2_bantuan8, text='Hidder Layer 1', background='#e2e6e2')
my_frame_2_text2.pack(side=tkinter.LEFT)
my_frame_2_text3 = tkinter.Label(my_frame_2_bantuan11, text='Nodes', background='#e2e6e2')
my_frame_2_text3.pack(side=tkinter.LEFT)
my_frame_2_text4 = tkinter.Label(my_frame_2_bantuan12, text='Nodes', background='#e2e6e2')
my_frame_2_text4.pack(side=tkinter.LEFT)
my_frame_2_text5 = tkinter.Label(my_frame_2_bantuan13, text='Hidden Layer 2', background='#e2e6e2')
my_frame_2_text5.pack(side=tkinter.LEFT)
my_frame_2_text6 = tkinter.Label(my_frame_2_bantuan14, text='Output Layer', background='#e2e6e2')
my_frame_2_text6.pack(side=tkinter.LEFT)
my_frame_2_text7 = tkinter.Label(my_frame_2_bantuan17, text='Nodes', background='#e2e6e2')
my_frame_2_text7.pack(side=tkinter.LEFT)
my_frame_2_text8 = tkinter.Label(my_frame_2_bantuan18, text='Nodes', background='#e2e6e2')
my_frame_2_text8.pack(side=tkinter.LEFT)
my_frame_2_text9 = tkinter.Label(my_frame_2_bantuan9, text=' = ', background='#e2e6e2')
my_frame_2_text9.pack(side=tkinter.LEFT)
my_frame_2_text10 = tkinter.Label(my_frame_2_bantuan10, text=' = ', background='#e2e6e2')
my_frame_2_text10.pack(side=tkinter.LEFT)
my_frame_2_text11 = tkinter.Label(my_frame_2_bantuan15, text=' = ', background='#e2e6e2')
my_frame_2_text11.pack(side=tkinter.LEFT)
my_frame_2_text12 = tkinter.Label(my_frame_2_bantuan16, text=' = ', background='#e2e6e2')
my_frame_2_text12.pack(side=tkinter.LEFT)
my_frame_2_spinbox1 = tkinter.Spinbox(my_frame_2_bantuan9, from_=None, to=None, width=4)
my_frame_2_spinbox1.configure(justify=tkinter.CENTER)
my_frame_2_spinbox1.pack(pady=1, side=tkinter.LEFT)
my_frame_2_spinbox2 = tkinter.Spinbox(my_frame_2_bantuan10, from_=None, to=None, width=4)
my_frame_2_spinbox2.configure(justify=tkinter.CENTER)
my_frame_2_spinbox2.pack(pady=1, side=tkinter.LEFT)
my_frame_2_spinbox3 = tkinter.Spinbox(my_frame_2_bantuan15, from_=None, to=None, width=4)
my_frame_2_spinbox3.configure(justify=tkinter.CENTER)
my_frame_2_spinbox3.pack(pady=1, side=tkinter.LEFT)
my_frame_2_spinbox4 = tkinter.Spinbox(my_frame_2_bantuan16, from_=None, to=None, width=4)
my_frame_2_spinbox4.configure(justify=tkinter.CENTER)
my_frame_2_spinbox4.pack(pady=1, side=tkinter.LEFT)

my_frame_3 = tkinter.LabelFrame(part_1, width=500, height=75, text='''Result''', relief='groove', foreground="black",
                                background='#e2e6e2')
my_frame_3.pack(padx=5, side=tkinter.LEFT)
my_frame_3.pack_propagate(0)
my_frame_3_bantuan1 = tkinter.Frame(my_frame_3, width=250, height=75, background='#e2e6e2')
my_frame_3_bantuan1.pack(side=tkinter.LEFT)
my_frame_3_bantuan1.pack_propagate(0)
my_frame_3_bantuan2 = tkinter.Frame(my_frame_3, width=250, height=75, background='#e2e6e2')
my_frame_3_bantuan2.pack(side=tkinter.LEFT)
my_frame_3_bantuan2.pack_propagate(0)
my_frame_3_text1 = tkinter.Label(my_frame_3_bantuan1, text='Iteration =', background='#e2e6e2')
my_frame_3_text1.pack(pady=4)
my_frame_3_text2 = tkinter.Label(my_frame_3_bantuan2, text='Error =', background='#e2e6e2')
my_frame_3_text2.pack(pady=4)
my_frame_3_entry1 = tkinter.Entry(my_frame_3_bantuan1, width=15)
my_frame_3_entry1.configure(justify=tkinter.CENTER)
my_frame_3_entry1.pack()
my_frame_3_entry2 = tkinter.Entry(my_frame_3_bantuan2, width=15)
my_frame_3_entry2.configure(justify=tkinter.CENTER)
my_frame_3_entry2.pack()

my_frame_4 = tkinter.Frame(part_2, width=103, height=625, background='#e2e6e2')
my_frame_4.pack(padx=5, pady=5, side=tkinter.LEFT)
my_frame_4.pack_propagate(0)

my_frame_5 = tkinter.Frame(part_2, width=400, height=625, background='#e2e6e2')
my_frame_5.pack(pady=5, side=tkinter.LEFT)
my_frame_5.pack_propagate(0)

my_frame_5_scrolledtext1 = tkst.ScrolledText(my_frame_5, wrap=tkinter.WORD, width=400, height=500,
                                                     background='white')
my_frame_5_scrolledtext1.pack(pady=5, fill=tkinter.BOTH, expand=True)
my_frame_5_scrolledtext1.configure(font="TkTextFont")
my_frame_5_scrolledtext1.configure(foreground="black")
my_frame_5_scrolledtext1.configure(highlightbackground="#e2e6e2")
my_frame_5_scrolledtext1.configure(highlightcolor="black")
my_frame_5_scrolledtext1.configure(insertbackground="black")
my_frame_5_scrolledtext1.configure(insertborderwidth="3")
my_frame_5_scrolledtext1.configure(selectbackground="#c4c4c4")
my_frame_5_scrolledtext1.configure(selectforeground="black")

class Plot:
    def __init__(self, id, x, y, label, kondisi, color):
        self.id = id
        self.x = x
        self.y = y
        self.label = label
        self.kondisi = kondisi
        self.color = color

class Axis:
    def __init__(self, top, judul, labelx, labely):
        self.master = top
        self.labelx = labelx
        self.labely = labely
        self.title = judul
        self.fig = Figure(figsize=(6, 1.1))
        self.fig.set_facecolor('#e2e6e2')
        self.grafik_windows = FigureCanvasTkAgg(self.fig, self.master)
        self.ax = self.fig.add_subplot(111)
        self.grafik_windows.get_tk_widget().pack()
        self.Attribute()
        box = self.ax.get_position()
        self.ax.set_position([box.x0 - box.width * 0.05, box.y0 + box.height * 0.18,
                              box.width * 0.9, box.height * 0.75])
        self.toolbar = NavigationToolbar2Tk(self.grafik_windows, self.master)
        self.toolbar.config(background='#e2e6e2')
        self.toolbar._message_label.config(background='#e2e6e2')
        self.toolbar.update()
        def on_key_press_1(event):
                print("you pressed {}".format(event.key))
                key_press_handler(event, self.grafik_windows, self.toolbar)
        self.grafik_windows.mpl_connect("key_press_event", on_key_press_1)

    def Attribute(self):
        self.ax.set_title(self.title, fontsize=8)
        self.ax.set_xlabel(self.labelx, fontsize=8)
        self.ax.set_ylabel(self.labely, fontsize=8)
        self.ax.tick_params(direction='in', labelsize=6)
        self.ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1.1), shadow=True, ncol=1)

    def draw_plot(self):
        self.ax.clear()
        self.Attribute()
        for item in self.plotlist:
                if item.kondisi == 0:
                        self.ax.plot(item.x, item.y, label=item.label, color=item.color, linewidth=0.5)
                elif item.kondisi == 1:
                        self.ax.bar(item.x, item.y, label=item.label, color=item.color, width=1)
        self.ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1.1), shadow=True, ncol=1)
        self.grafik_windows.draw()

    def plot(self, x, y, label, kondisi=0, color='blue'):
        self.plotlist = []
        self.add_plot(0, x, y, label, kondisi, color)

    def add_plot(self, id, x, y, label, kondisi=0, color='blue'):
        for i, item in enumerate(self.plotlist):  # Jika ada plot dengan id yg sama
                if item.id == id:
                        self.plotlist[i] = Plot(id, x, y, label, kondisi, color)
                        self.draw_plot()
                        return

        self.plotlist += [Plot(id, x, y, label, kondisi, color)]  # Jika belum ada plot dengan id yang ditentukan
        self.draw_plot()

    def clearfig(self):
        self.ax.clear()
        self.Attribute()
        self.grafik_windows.draw()

my_frame_6 = tkinter.Frame(part_2, width=500, height=625, background='#e2e6e2')
my_frame_6.pack(padx=5, pady=5, side=tkinter.LEFT)
my_frame_6.pack_propagate(0)
my_frame_6_bantuan1 = tkinter.Frame(my_frame_6, width=500, height=150, background='#e2e6e2')
my_frame_6_bantuan1.pack()
my_frame_6_bantuan1.pack_propagate(0)
my_frame_6_bantuan2 = tkinter.Frame(my_frame_6, width=500, height=325, background='#e2e6e2')
my_frame_6_bantuan2.pack()
my_frame_6_bantuan2.pack_propagate(0)
my_frame_6_bantuan3 = tkinter.Frame(my_frame_6, width=500, height=150, background='#e2e6e2')
my_frame_6_bantuan3.pack()
my_frame_6_bantuan3.pack_propagate(0)
my_frame_6_ax1 = Axis(my_frame_6_bantuan1, "Input Data", 'n', 'A')
my_frame_6_ax2 = Axis(my_frame_6_bantuan3, "Error", 'Iteration', 'A')
style = ttk.Style()
style.configure('style.TFrame', background='#e2e6e2', borderwidth=0, foreground='#e2e6e2',
                highlightbackground='#e2e6e2', highlightthickness=0)
my_frame_6_bantuan2_control = ttk.Notebook(my_frame_6_bantuan2, width=500, height=300, style='style.TFrame')
my_frame_6_bantuan2_control.pack()
my_frame_6_bantuan2_control.pack_propagate(0)

def tabNotebook_init():
    global my_frame_6_bantuan2_output_list, my_frame_6_bantuan2_axis_list, my_frame_6_ax

    for i, item in enumerate(my_frame_6_bantuan2_control.winfo_children()):
        item.destroy()
    my_frame_6_bantuan2_output_list = []
    my_frame_6_bantuan2_output_list += [tkinter.Frame(my_frame_6_bantuan2_control, width=500, height=325)]
    my_frame_6_bantuan2_output_list[0].configure()
    my_frame_6_bantuan2_control.add(my_frame_6_bantuan2_output_list[0], text='Node 1')
    my_frame_6_bantuan2_axis_list = []
    my_frame_6_bantuan2_axis_list += [(tkinter.Frame(my_frame_6_bantuan2_output_list[0], width=500, height=155,
                                                    background='#e2e6e2'),
                                      tkinter.Frame(my_frame_6_bantuan2_output_list[0], width=500, height=155,
                                                    background='#e2e6e2'))]
    my_frame_6_bantuan2_axis_list[0][0].pack()
    my_frame_6_bantuan2_axis_list[0][0].pack_propagate(0)
    my_frame_6_bantuan2_axis_list[0][1].pack()
    my_frame_6_bantuan2_axis_list[0][1].pack_propagate(0)
    my_frame_6_ax = []
    my_frame_6_ax += [(Axis(my_frame_6_bantuan2_axis_list[0][0], "Target", 'n', 'A'),
                      Axis(my_frame_6_bantuan2_axis_list[0][1], "Recall", 'n', 'A'))]
tabNotebook_init()

# Main Program
# Variable Declaration
input_status, topology_status, initial_status, training_status, recall_status = 0, 0, 0, 0, 0
alpha1 = 0.145
TD_Input = [] # Training Data Input
TD_Target = [] # Training Data Target

def scrolledtext_init():
    my_frame_5_scrolledtext1.delete('1.0', tkinter.END)
    my_frame_5_scrolledtext1.insert(tkinter.INSERT, """Welcome to Multilayer Perceptron Network with EBPA Program \
Created by Dzikrur Rohmani Z R M H, BME Dept. ITS\n\nPlease choose training data that you want to learn!\n\n""")

    if input_status == 1:
        my_frame_5_scrolledtext1.insert(tkinter.INSERT, "{} data to be learned (I=Input node, T=Target node):\n".format(
            training_data))
        for i in range(data_masuk):
            my_frame_5_scrolledtext1.insert(tkinter.INSERT,"I_{}\t".format(i+1))
        for i in range(target_masuk):
            if i<target_masuk-1: my_frame_5_scrolledtext1.insert(tkinter.INSERT,"T_{}\t".format(i+1))
            else: my_frame_5_scrolledtext1.insert(tkinter.INSERT,"T_{}\n".format(i+1))
        for i in range(int(datanum)):
            for j in range(data_masuk):
                my_frame_5_scrolledtext1.insert(tkinter.INSERT, "{}\t".format(TD_Input[i][j]))
            for j in range(target_masuk):
                if j<target_masuk-1: my_frame_5_scrolledtext1.insert(tkinter.INSERT, "{}\t".format(TD_Target[i][j]))
                else: my_frame_5_scrolledtext1.insert(tkinter.INSERT, "{}\n".format(TD_Target[i][j]))
        my_frame_5_scrolledtext1.insert(tkinter.INSERT, "\nPlease continue to network design or recall the optimized \
network's parameter of {} training data in the past!\n\n".format(training_data))

    if topology_status == 1:
        my_frame_5_scrolledtext1.insert(tkinter.INSERT, """ANN Topology:\nInput Layer = {} Nodes\nHidden Layer 1 = {} \
Nodes\nHidden Layer 2 = {} Nodes\nOutput Layer = {} Nodes\n\nPlease continue with initialization...\
!\n\n""".format(Layer1, Layer2, Layer3, Layer4))

    if initial_status == 1:
        my_frame_5_scrolledtext1.insert(tkinter.INSERT, """Weight initialization\nWeight Input Layer to Hidden Layer 1\
({}x{}) =\n{}\n\nWeight Hidden Layer 1 to Hidden Layer 2({}x{}) =\n{}\n\nWeight Hidden Layer 2 to Output Layer 1\
({}x{}) =\n{}\n\nNetwork ready to have a learning session..\n\n""".format(Layer2, Layer1, W1_2, Layer3, Layer2,
                                                                                  W2_3, Layer4, Layer3, W3_4))

    if training_status == 1:
        my_frame_5_scrolledtext1.insert(tkinter.INSERT, "The optimized network's parameter of {} training data already \
saved into a file and ready to be tested..\nPlease do a recall sesion!\n\n".format(training_data))

    if recall_status == 1:
        my_frame_5_scrolledtext1.insert(tkinter.INSERT, "The optimized network's parameter of {} training data from {} \
iteration with {:.10f} sum squarred error in the past has recalled succesfully..\nANN Topology:\nInputLayer = {} Nodes\n\
Hidden Layer 1 = {} Nodes\nHidden Layer 2 = {} Nodes\nOutput Layer = {} Nodes\n\nOptimal connection weights :\nWeight \
Input Layer to Hidden Layer 1 ({}x{}) =\n{}\n\nWeight Hidden Layer 1 to Hidden Layer 2({}x{}) =\n{}\n\nWeight Hidden \
Layer 2 to Output Layer 1({}x{}) =\n{}\n\nRecall output (R = Recall node):\n".format(training_data,
                                                                 int(baca_Iteration), baca_Error,
                                                                 baca_Layer1, baca_Layer2, baca_Layer3, baca_Layer4,
                                                                 baca_Layer2, baca_Layer1, baca_W1_2, baca_Layer3,
                                                                 baca_Layer2, baca_W2_3, baca_Layer4, baca_Layer3,
                                                                 baca_W3_4))
        for i in range(baca_Layer4):
            if i < baca_Layer4-1: my_frame_5_scrolledtext1.insert(tkinter.INSERT, "R_{}\t".format(i+1))
            else: my_frame_5_scrolledtext1.insert(tkinter.INSERT, "R_{}\n".format(i+1))
        for i in range(int(datanum)):
            for j in range(baca_Layer4):
                if j < baca_Layer4-1: my_frame_5_scrolledtext1.insert(tkinter.INSERT,
                                                                      "{:.2f}\t".format(Final_Output[j][i]))
                else: my_frame_5_scrolledtext1.insert(tkinter.INSERT, "{:.2f}\n".format(Final_Output[j][i]))
        my_frame_5_scrolledtext1.insert(tkinter.INSERT, "\nRecall absolut error (E = Recall absolute error):"
                                                        "\n".format(i+1))
        for i in range(baca_Layer4):
            if i < baca_Layer4-1: my_frame_5_scrolledtext1.insert(tkinter.INSERT, "E_{}\t".format(i+1))
            else: my_frame_5_scrolledtext1.insert(tkinter.INSERT, "E_{}\n".format(i+1))
        for i in range(int(datanum)):
            for j in range(baca_Layer4):
                if j < baca_Layer4-1: my_frame_5_scrolledtext1.insert(tkinter.INSERT,
                                                                      "{:.2f}\t".format(Error_Output[j][i]))
                else: my_frame_5_scrolledtext1.insert(tkinter.INSERT, "{:.2f}\n".format(Error_Output[j][i]))

        Ave_E = 0
        for i in range(baca_Layer4):
            Ave_E += sum(Error_Output[i])/(datanum*baca_Layer4)
        my_frame_5_scrolledtext1.insert(tkinter.INSERT, "\nAverage Error = {}\n\n\
Congratulation.. Network has been tested successfully..\n\n".format(Ave_E))

scrolledtext_init()

def spin_init():
    my_frame_2_spinbox1.delete(0, tkinter.END)
    my_frame_2_spinbox2.delete(0, tkinter.END)
    my_frame_2_spinbox3.delete(0, tkinter.END)
    my_frame_2_spinbox4.delete(0, tkinter.END)

def spin_update():
    my_frame_2_spinbox1.delete(0, tkinter.END)
    my_frame_2_spinbox1.insert(0, data_masuk)
    my_frame_2_spinbox2.__setitem__('to', 10)
    my_frame_2_spinbox2.__setitem__('from_', 2)
    my_frame_2_spinbox2.delete(0, tkinter.END)
    my_frame_2_spinbox2.insert(0, 2)
    my_frame_2_spinbox3.__setitem__('to', 10)
    my_frame_2_spinbox3.__setitem__('from_', 2)
    my_frame_2_spinbox3.delete(0, tkinter.END)
    my_frame_2_spinbox3.insert(0, 2)
    my_frame_2_spinbox4.delete(0, tkinter.END)
    my_frame_2_spinbox4.insert(0, target_masuk)

def set_input_data():
    global datanum, data_masuk, target_masuk, TD_Input, TD_Target, training_data, input_status, \
        topology_status, initial_status, training_status, my_frame_6_bantuan2_output_list, \
        my_frame_6_bantuan2_axis_list, my_frame_6_ax

    if recall_status == 1:
        mb.showerror("Error", "The network has completed the task..\nPlease clear before doing next task!")
        return

    training_data = inputan.get()
    try:
        f = open("{}.dat".format(training_data), "r")
        read_data_training = f.readlines()
        In_LF = np.zeros([int(read_data_training[0]),2])
        Out_LF = np.zeros([int(read_data_training[0]),np.size(read_data_training[1].split("\t"))-2])
        for i in range(int(read_data_training[0])):
            temp = read_data_training[i+1].split("\t")
            for j in range(2):
                In_LF[i][j] = temp[j]
            for j in range(np.size(temp)-2):
                Out_LF[i][j] = temp[j+2]
    except:
        mb.showerror("Error", "Please choose training data that you want to learn!")
        input_status = 0
        return

    spin_init()
    data_masuk = np.size(In_LF[0])
    target_masuk = np.size(Out_LF[0])
    datanum = np.size(In_LF)/data_masuk
    TD_Input = In_LF
    TD_Target = Out_LF
    spin_update()

    Input_Plot_List = []
    for i in range(data_masuk):
        temp = []
        for j in range(int(datanum)):
            temp += [TD_Input[j][i]]
        Input_Plot_List += [(temp)]

    tabNotebook_init()
    my_frame_6_ax1.plot(np.arange(1, datanum+1), Input_Plot_List[0], 'X1', 0, 'blue')
    my_frame_6_ax1.add_plot(1, np.arange(1, datanum+1), Input_Plot_List[1], 'X2', 0, 'red')

    for i in range(1, target_masuk):
        my_frame_6_bantuan2_output_list += [ttk.Frame(my_frame_6_bantuan2_control, width=600, height=325,
                                                      style='style.TFrame')]
        my_frame_6_bantuan2_control.add(my_frame_6_bantuan2_output_list[i], text='Node {}'.format(i+1))
        my_frame_6_bantuan2_axis_list += [(tkinter.Frame(my_frame_6_bantuan2_output_list[i], width=500, height=155,
                                                        background='#e2e6e2'),
                                          tkinter.Frame(my_frame_6_bantuan2_output_list[i], width=500, height=155,
                                                        background='#e2e6e2'))]
        my_frame_6_bantuan2_axis_list[i][0].pack()
        my_frame_6_bantuan2_axis_list[i][0].pack_propagate(0)
        my_frame_6_bantuan2_axis_list[i][1].pack()
        my_frame_6_bantuan2_axis_list[i][1].pack_propagate(0)
        my_frame_6_ax += [(Axis(my_frame_6_bantuan2_axis_list[i][0], "Target", 'n', 'A'),
                          Axis(my_frame_6_bantuan2_axis_list[i][1], "Recall", 'n', 'A'))]

    for i in range(target_masuk):
        temp = []
        for j in range(int(datanum)):
            temp += [TD_Target[j][i]]
        my_frame_6_ax[i][0].plot(np.arange(1, datanum + 1), temp, 'Target', 1, 'blue')

    input_status = 1
    topology_status, initial_status, training_status = 0, 0, 0
    scrolledtext_init()

def ANN_Topology():
    global Layer1, Layer2, Layer3, Layer4, topology_status, initial_status, training_status

    if recall_status == 1:
        mb.showerror("Error", "The network has completed the task..\nPlease clear before doing next task!")
        return

    if input_status == 0:
        mb.showerror("Error", "Please set training data that you want to learn!")
        return

    Layer1 = int(my_frame_2_spinbox1.get())
    Layer2 = int(my_frame_2_spinbox2.get())
    Layer3 = int(my_frame_2_spinbox3.get())
    Layer4 = int(my_frame_2_spinbox4.get())

    topology_status = 1
    initial_status, training_status = 0,0
    scrolledtext_init()

def weight_initialization():
    global W1_2, W2_3, W3_4, Th, initial_status, training_status

    if recall_status == 1:
        mb.showerror("Error", "The network has completed the task..\nPlease clear before doing next task!")
        return

    if topology_status == 0:
        mb.showerror("Error", "Please do network topology design before initializing weight!")
        return

    W1_2 = np.zeros([Layer1, Layer2])  # Weight Input Layer to Hidden Layer 1
    W2_3 = np.zeros([Layer2, Layer3])  # Weight Hidden Layer 1 to Hidden Layer 2
    W3_4 = np.zeros([Layer3, Layer4])  # Weight Hidden Layer 2 to Output Layer

    for i in range(Layer1):
        for j in range(Layer2):
            W1_2[i][j] = np.random.normal(0, 0.5)
    for i in range(Layer2):
        for j in range(Layer3):
            W2_3[i][j] = np.random.normal(0, 0.5)
    for i in range(Layer3):
        for j in range(Layer4):
            W3_4[i][j] = np.random.normal(0, 0.5)

    Th = np.zeros([Layer2+Layer3+Layer4])
    for i in range(int(np.size(Th))):
        Th[i] = 0.1

    initial_status = 1
    training_status = 0
    scrolledtext_init()

def training_process():
    global training_status, Iteration, E, initial_status, topology_status

    if recall_status == 1:
        mb.showerror("Error", "The network has completed the task..\nPlease clear before doing next task!")
        return

    if initial_status == 0:
        mb.showerror("Error","Please initializing random weight before do training process!")
        return

    All_E = []
    Input = np.zeros([Layer1])
    OutHidden1 = np.zeros([Layer2])
    OutHidden2 = np.zeros([Layer3])
    OutLayer = np.zeros([Layer4])

    Error_limit = 0.3
    Iteration = 0 # counter for iteration
    start = datetime.now()
    while True:
        E = 0
        counter = 0
        while counter < datanum:
            # Layer 1
            for j in range(Layer1):
                Input[j] = TD_Input[counter][j]

            # Layer 2
            for j in range(Layer2):
                temp = 0.0
                for k in range(Layer1):
                    temp += Input[k]*W1_2[k][j]
                temp -= Th[j]
                OutHidden1[j] = 1/(1+np.exp(-temp))

            # Layer 3
            for j in range(Layer3):
                temp = 0.0
                for k in range(Layer2):
                    temp += OutHidden1[k] * W2_3[k][j]
                temp -= Th[j+Layer2]
                OutHidden2[j] = 1 / (1 + np.exp(-temp))

            # Layer 4
            for j in range(Layer4):
                temp = 0.0
                for k in range(Layer3):
                    temp += OutHidden2[k] * W3_4[k][j]
                temp -= Th[j+Layer2+Layer3]
                OutLayer[j] = 1 / (1 + np.exp(-temp))

            e_node_4 = np.zeros([Layer4])
            for j in range(Layer4):
                e_node_4[j] = TD_Target[counter][j]-OutLayer[j]  # output node error
                E += 0.5*(e_node_4[j]**2)  # SSE (weigthed sum squarred error)
            All_E += [E]

            # Update W3_4
            del_OutLayer = np.zeros(Layer4)
            for j in range(Layer4):
                del_OutLayer[j] = e_node_4[j]*OutLayer[j]*(1-OutLayer[j])  # Equation 3.56
                Th[Layer2 + Layer3 + j] -= alpha1 * del_OutLayer[j]

            e_node_3 = np.zeros([Layer3])
            for j in range(Layer3):
                for k in range(Layer4):
                    e_node_3[j] += W3_4[j][k]*e_node_4[k]

            del_HiddenLayer2 = np.zeros([Layer3])
            for j in range(Layer3):
                del_HiddenLayer2[j] = e_node_3[j]*OutHidden2[j]*(1-OutHidden2[j])  # Equation 3.57
                Th[Layer2 + j] -= alpha1 * del_HiddenLayer2[j]

            # Weight updating
            for j in range(Layer3):
                for k in range(Layer4):
                    W3_4[j][k] += alpha1*del_OutLayer[k]*OutHidden2[j]

            # Update W2_3
            e_node_2 = np.zeros([Layer2])
            for j in range(Layer2):
                for k in range(Layer3):
                    e_node_2[j] += W2_3[j][k] * e_node_3[k]

            del_HiddenLayer1 = np.zeros([Layer2])
            for j in range(Layer2):
                del_HiddenLayer1[j] = e_node_2[j] * OutHidden1[j] * (1 - OutHidden1[j])  # Equation 3.57
                Th[j] -= alpha1 * del_HiddenLayer1[j]


            # Weight updating
            for j in range(Layer2):
                for k in range(Layer3):
                    W2_3[j][k] += alpha1 * del_HiddenLayer2[k] * OutHidden1[j]

            # Update W1_2
            # Weight updating
            for j in range(Layer1):
                for k in range(Layer2):
                    W1_2[j][k] += alpha1 * del_HiddenLayer1[k] * Input[j]

            counter += 1
            Iteration += 1
        if E < Error_limit:
            break

    end = datetime.now()
    time_taken = end - start
    print('Time: ', time_taken)

    Simpan_properties = open("{}_Properties.txt".format(training_data),"w")
    properties = "{}\tInput\n{}\tInput\n{}\tInput\n{}\tInput\n{}\tIteration\n{}\tSSE\n{}\tTime taken".format(Layer1,
                                                                                             Layer2, Layer3,
                                                                                             Layer4, Iteration,
                                                                                             E, time_taken)
    Simpan_properties.write(properties)
    Simpan_properties.close()

    Simpan_W1_2 = open("{}_W1_2.txt".format(training_data), "w")
    for j in range(Layer1):
        for k in range(Layer2):
            Simpan_W1_2.write("{}\t".format(W1_2[j][k]))
        Simpan_W1_2.write("\n".format(W1_2[j][k]))
    Simpan_W1_2.close()

    Simpan_W2_3 = open("{}_W2_3.txt".format(training_data), "w")
    for j in range(Layer2):
        for k in range(Layer3):
            Simpan_W2_3.write("{}\t".format(W2_3[j][k]))
        Simpan_W2_3.write("\n".format(W2_3[j][k]))
    Simpan_W2_3.close()

    Simpan_W3_4 = open("{}_W3_4.txt".format(training_data), "w")
    for j in range(Layer3):
        for k in range(Layer4):
            Simpan_W3_4.write("{}\t".format(W3_4[j][k]))
        Simpan_W3_4.write("\n".format(W3_4[j][k]))
    Simpan_W3_4.close()

    Simpan_Th = open("{}_Th.txt".format(training_data), "w")
    for j in range(Layer2+Layer3+Layer4):
        Simpan_Th.write("{}\t".format(Th[j]))
    Simpan_Th.close()

    Simpan_E = open("{}_E.txt".format(training_data), "w")
    for j in range(Iteration):
        Simpan_E.write("{}\n".format(All_E[j]))
    Simpan_E.close()

    training_status = 1
    topology_status, initial_status = 0, 0
    scrolledtext_init()
    mb.showinfo("Message","The optimized network's parameter of {} training data from {} iteration with {:.10f} sum \
squarred error already saved into a file and ready to be tested..\nPlease do a recall sesion!".format(training_data,
                                                                                                      Iteration, E))

def Recall():
    global baca_Layer1, baca_Layer2, baca_Layer3, baca_Layer4, baca_Iteration, baca_Error, baca_W1_2, baca_W2_3, \
        baca_W3_4, topology_status, initial_status, recall_status, Final_Output, Error_Output

    if input_status == 0:
        mb.showerror("Error", "Please set training data that you want to learn!")
        return

    A = open("{}_Properties.txt".format(training_data),"r")
    Baca_properties = A.readlines()

    prop1 = []
    for item in Baca_properties:
        prop2 = item.split('\t')
        try:
            prop1 += [float(prop2[0])]
        except:
            prop1 += [prop2[0]]
    A.close()
    baca_Layer1 = int(prop1[0])
    baca_Layer2 = int(prop1[1])
    baca_Layer3 = int(prop1[2])
    baca_Layer4 = int(prop1[3])
    baca_Iteration = prop1[4]
    baca_Error = prop1[5]

    baca_W1_2 = np.zeros([baca_Layer1, baca_Layer2])
    baca_W2_3 = np.zeros([baca_Layer2, baca_Layer3])
    baca_W3_4 = np.zeros([baca_Layer3, baca_Layer4])
    B = open("{}_W1_2.txt".format(training_data), "r")
    Baca_W1_2 = B.readlines()
    for j in range(baca_Layer1):
        W1 = Baca_W1_2[j].split('\t')
        for k in range(baca_Layer2):
            baca_W1_2[j][k] = W1[k]
    B.close()

    C = open("{}_W2_3.txt".format(training_data), "r")
    Baca_W2_3 = C.readlines()
    for j in range(baca_Layer2):
        W2 = Baca_W2_3[j].split('\t')
        for k in range(baca_Layer3):
            baca_W2_3[j][k] = W2[k]
    C.close()

    D = open("{}_W3_4.txt".format(training_data), "r")
    Baca_W3_4 = D.readlines()
    for j in range(baca_Layer3):
        W3 = Baca_W3_4[j].split('\t')
        for k in range(baca_Layer4):
            baca_W3_4[j][k] = W3[k]
    D.close()

    E = open("{}_Th.txt".format(training_data), "r")
    Baca_Th = E.readlines()
    for item in Baca_Th:
        baca_Th = item.split('\t')
    E.close()

    F = open("{}_E.txt".format(training_data), "r")
    Baca_All_E = F.readlines()
    baca_All_E = np.zeros([int(baca_Iteration)])
    for j in range(int(baca_Iteration)):
        baca_All_E[j] = Baca_All_E[j]
    F.close()

    counter = 0
    Final_Input = np.zeros([baca_Layer1])
    Final_OutHidden1 = np.zeros([baca_Layer2])
    Final_OutHidden2 = np.zeros([baca_Layer3])
    Final_Output, Error_Output = [], []
    fix_out_counter, fix_error_counter = [], []
    while counter<datanum:
        # Layer 1
        for j in range(baca_Layer1):
            Final_Input[j] = TD_Input[counter][j]

        # Layer 2
        for j in range(baca_Layer2):
            temp = 0.0
            for k in range(baca_Layer1):
                temp += Final_Input[k]*baca_W1_2[k][j]
            temp -= float(baca_Th[j])
            Final_OutHidden1[j] = 1/(1+np.exp(-temp))

        # Layer 3
        for j in range(baca_Layer3):
            temp = 0.0
            for k in range(baca_Layer2):
                temp += Final_OutHidden1[k] * baca_W2_3[k][j]
            temp -= float(baca_Th[j+baca_Layer2])
            Final_OutHidden2[j] = 1 / (1 + np.exp(-temp))

        # Layer 4
        for j in range(baca_Layer4):
            temp = 0.0
            for k in range(baca_Layer3):
                temp += Final_OutHidden2[k] * baca_W3_4[k][j]
            temp -= float(baca_Th[j+baca_Layer2+baca_Layer3])
            fix_out_counter += [1/(1+np.exp(-temp))]
            fix_error_counter += [abs(TD_Target[counter][j] - (1/(1+np.exp(-temp))))]
        counter += 1

    for i in range(baca_Layer4):
        tempa, tempb = [], []
        for j in range(int(datanum)):
            tempa += [fix_out_counter[int((baca_Layer4*j)+i)]]
            tempb += [fix_error_counter[int((baca_Layer4*j)+i)]]
        Final_Output += [(tempa)]
        Error_Output += [(tempb)]

    my_frame_3_entry1.delete(0, tkinter.END)
    my_frame_3_entry2.delete(0, tkinter.END)
    my_frame_3_entry1.insert(0, int(baca_Iteration))
    my_frame_3_entry2.insert(0, "{:.10f}".format(baca_Error))

    for i in range(baca_Layer4):
        my_frame_6_ax[i][1].plot(np.arange(1, datanum + 1), Final_Output[i], 'Target', 1, 'blue')
    my_frame_6_ax2.plot(np.arange(1,baca_Iteration+1), baca_All_E, 'Error', 0, 'black')

    recall_status = 1
    topology_status, initial_status = 0, 0
    scrolledtext_init()

def Clear_all():
    global input_status, recall_status, topology_status, initial_status

    spin_init()
    input_status, recall_status, topology_status, initial_status = 0, 0, 0, 0
    my_frame_1_combobox.delete(0, tkinter.END)
    my_frame_6_ax1.clearfig()
    my_frame_6_ax2.clearfig()
    tabNotebook_init()
    scrolledtext_init()
    my_frame_3_entry1.delete(0, tkinter.END)
    my_frame_3_entry2.delete(0, tkinter.END)

def About():
    mb.showinfo("About", "Multilayer Perceptron Network Model using Error Back Propagation Algorithm \
for Gait Phase Data Recognition by Dzikrur Rohmani Z R M H.\n\nBiomedical Engineering Department\nInstitut Teknologi \
Sepuluh Nopember (ITS) Surabaya")

def Close():
    windows.quit()
    # windows.destroy()

# Command Function
my_frame_1_button = tkinter.Button(my_frame_1, text='SET', width=9, command=set_input_data)
my_frame_1_button.pack()
my_frame_4_button1 = tkinter.Button(my_frame_4, width=13, text='Topology Design', command=ANN_Topology)
my_frame_4_button1.pack(pady=5)
my_frame_4_button2 = tkinter.Button(my_frame_4, width=13, text='Initialization', command=weight_initialization)
my_frame_4_button2.pack(pady=5)
my_frame_4_button3 = tkinter.Button(my_frame_4, width=13, text='ANN Training', command=training_process)
my_frame_4_button3.pack(pady=5)
my_frame_4_button4 = tkinter.Button(my_frame_4, width=13, text='Recall', command=Recall)
my_frame_4_button4.pack(pady=5)
my_frame_4_button5 = tkinter.Button(my_frame_4, width=13, text='Clear', command=Clear_all)
my_frame_4_button5.pack(pady=5)
my_frame_4_button6 = tkinter.Button(my_frame_4, width=13, text='About', command=About)
my_frame_4_button6.pack(pady=5)
my_frame_4_button7 = tkinter.Button(my_frame_4, width=13, text='Close', command=Close)
my_frame_4_button7.pack(pady=5)

windows.mainloop()
