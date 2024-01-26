import customtkinter
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import numpy as np

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


class Arcball(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        # Orientation vars. Initialized to represent 0 rotation
        self.quat = np.array([[1],[0],[0],[0]])
        self.rotM = np.eye(3)
        self.AA = {"axis": np.array([[0],[0],[0]]), "angle":0.0}
        self.rotv = np.array([[0],[0],[0]])
        self.euler = np.array([[0],[0],[0]])

        # configure window
        self.title("Holroyd's arcball")
        self.geometry(f"{1100}x{580}")
        self.resizable(False, False)

        self.grid_columnconfigure((0,1), weight=0)
        self.grid_rowconfigure((0,1), weight=1)
        self.grid_rowconfigure(2, weight=0)

        # Cube plot
        self.init_cube()

        self.canvas = FigureCanvasTkAgg(self.fig, self)  # A tk.DrawingArea.
        self.bm = BlitManager(self.canvas,[self.facesObj])
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.pressed = False #Bool to bypass the information that mouse is clicked
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        
        # Reset button
        self.resetbutton = customtkinter.CTkButton(self, text="Reset", command=self.resetbutton_pressed)
        self.resetbutton.grid(row=3, column=0, padx=(0, 0), pady=(5, 20), sticky="ns")
        
        # Selectable atti
        self.tabview = customtkinter.CTkTabview(self, width=150, height=150)
        self.tabview.grid(row=0, column=1, padx=(0, 20), pady=(20, 0), sticky="nsew")
        self.tabview.add("Axis angle")
        self.tabview.add("Rotation vector")
        self.tabview.add("Euler angles")
        self.tabview.add("Quaternion")

        # Selectable atti: AA
        self.tabview.tab("Axis angle").grid_columnconfigure(0, weight=0)  # configure grid of individual tabs
        self.tabview.tab("Axis angle").grid_columnconfigure(1, weight=0)  # configure grid of individual tabs

        self.label_AA_axis= customtkinter.CTkLabel(self.tabview.tab("Axis angle"), text="Axis:")
        self.label_AA_axis.grid(row=0, column=0, rowspan=3, padx=(80,0), pady=(45,0), sticky="e")

        self.entry_AA_ax1 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax1.insert(0,"1.0")
        self.entry_AA_ax1.grid(row=0, column=1, padx=(5, 0), pady=(50, 0), sticky="ew")

        self.entry_AA_ax2 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax2.insert(0,"0.0")
        self.entry_AA_ax2.grid(row=1, column=1, padx=(5, 0), pady=(5, 0), sticky="ew")

        self.entry_AA_ax3 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax3.insert(0,"0.0")
        self.entry_AA_ax3.grid(row=2, column=1, padx=(5, 0), pady=(5, 10), sticky="ew")

        self.label_AA_angle = customtkinter.CTkLabel(self.tabview.tab("Axis angle"), text="Angle(rads):")
        self.label_AA_angle.grid(row=3, column=0, padx=(120,0), pady=(10, 20),sticky="w")
        self.entry_AA_angle = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_angle.insert(0,"0.0")
        self.entry_AA_angle.grid(row=3, column=1, padx=(5, 0), pady=(0, 10), sticky="ew")

        self.button_AA = customtkinter.CTkButton(self.tabview.tab("Axis angle"), text="Apply", command=self.apply_AA, width=180)
        self.button_AA.grid(row=5, column=0, columnspan=2, padx=(0, 0), pady=(5, 0), sticky="e")

        # Selectable atti: rotV
        self.tabview.tab("Rotation vector").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Rotation vector").grid_columnconfigure(1, weight=0)
        
        self.label_rotV= customtkinter.CTkLabel(self.tabview.tab("Rotation vector"), text="rot. Vector:")
        self.label_rotV.grid(row=0, column=0, rowspan=3, padx=(2,0), pady=(45,0), sticky="e")

        self.entry_rotV_1 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_1.insert(0,"0.0")
        self.entry_rotV_1.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_rotV_2 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_2.insert(0,"0.0")
        self.entry_rotV_2.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_rotV_3 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_3.insert(0,"0.0")
        self.entry_rotV_3.grid(row=2, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_rotV = customtkinter.CTkButton(self.tabview.tab("Rotation vector"), text="Apply", command=self.apply_rotV, width=180)
        self.button_rotV.grid(row=5, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Selectable atti: Euler angles
        self.tabview.tab("Euler angles").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Euler angles").grid_columnconfigure(1, weight=0)
        
        self.label_EA_roll= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="roll:")
        self.label_EA_roll.grid(row=0, column=0, padx=(2,0), pady=(50,0), sticky="e")

        self.label_EA_pitch= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="pitch:")
        self.label_EA_pitch.grid(row=1, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_EA_yaw= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="yaw:")
        self.label_EA_yaw.grid(row=2, column=0, rowspan=3, padx=(2,0), pady=(5,10), sticky="e")

        self.entry_EA_roll = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_roll.insert(0,"0.0")
        self.entry_EA_roll.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_EA_pitch = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_pitch.insert(0,"0.0")
        self.entry_EA_pitch.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_EA_yaw = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_yaw.insert(0,"0.0")
        self.entry_EA_yaw.grid(row=2, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_EA = customtkinter.CTkButton(self.tabview.tab("Euler angles"), text="Apply", command=self.apply_EA, width=180)
        self.button_EA.grid(row=5, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Selectable atti: Quaternion
        self.tabview.tab("Quaternion").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Quaternion").grid_columnconfigure(1, weight=0)
        
        self.label_quat_0= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q0:")
        self.label_quat_0.grid(row=0, column=0, padx=(2,0), pady=(50,0), sticky="e")

        self.label_quat_1= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q1:")
        self.label_quat_1.grid(row=1, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_quat_2= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q2:")
        self.label_quat_2.grid(row=2, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_quat_3= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q3:")
        self.label_quat_3.grid(row=3, column=0, padx=(2,0), pady=(5,10), sticky="e")

        self.entry_quat_0 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_0.insert(0,"1.0")
        self.entry_quat_0.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_quat_1 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_1.insert(0,"0.0")
        self.entry_quat_1.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_quat_2 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_2.insert(0,"0.0")
        self.entry_quat_2.grid(row=2, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_quat_3 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_3.insert(0,"0.0")
        self.entry_quat_3.grid(row=3, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_quat = customtkinter.CTkButton(self.tabview.tab("Quaternion"), text="Apply", command=self.apply_quat, width=180)
        self.button_quat.grid(row=4, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Rotation matrix info
        self.RotMFrame = customtkinter.CTkFrame(self, width=150)
        self.RotMFrame.grid(row=1, column=1, rowspan=3, padx=(0, 20), pady=(20, 20), sticky="nsew")

        self.RotMFrame.grid_columnconfigure((0,1,2,3,4), weight=1)

        self.label_RotM= customtkinter.CTkLabel(self.RotMFrame, text="RotM = ")
        self.label_RotM.grid(row=0, column=0, rowspan=3, padx=(2,0), pady=(20,0), sticky="e")

        self.entry_RotM_11= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_11.insert(0,"1.0")
        self.entry_RotM_11.configure(state="disabled")
        self.entry_RotM_11.grid(row=0, column=1, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_12= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_12.insert(0,"0.0")
        self.entry_RotM_12.configure(state="disabled")
        self.entry_RotM_12.grid(row=0, column=2, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_13= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_13.insert(0,"0.0")
        self.entry_RotM_13.configure(state="disabled")
        self.entry_RotM_13.grid(row=0, column=3, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_21= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_21.insert(0,"0.0")
        self.entry_RotM_21.configure(state="disabled")
        self.entry_RotM_21.grid(row=1, column=1, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_22= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_22.insert(0,"1.0")
        self.entry_RotM_22.configure(state="disabled")
        self.entry_RotM_22.grid(row=1, column=2, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_23= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_23.insert(0,"0.0")
        self.entry_RotM_23.configure(state="disabled")
        self.entry_RotM_23.grid(row=1, column=3, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_31= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_31.insert(0,"0.0")
        self.entry_RotM_31.configure(state="disabled")
        self.entry_RotM_31.grid(row=2, column=1, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_32= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_32.insert(0,"0.0")
        self.entry_RotM_32.configure(state="disabled")
        self.entry_RotM_32.grid(row=2, column=2, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_33= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_33.insert(0,"1.0")
        self.entry_RotM_33.configure(state="disabled")
        self.entry_RotM_33.grid(row=2, column=3, padx=(2,0), pady=(2,0), sticky="ew")

    def Eaa2rotM(self, angle, axis):
        '''
        Devuelve la matriz de rotación R capaz de rotar vectores un ángulo 'angle' (en radiane*s) alrededor del eje 'axis'
        '''
        axis = axis / np.linalg.norm(axis,2)#Normalizar los ejes

        R = np.zeros((3,3))

        uuT = np.array([[axis[0] * axis[0], axis[0] * axis[1], axis[0] * axis[2]],
                        [axis[1] * axis[0], axis[1] * axis[1], axis[1] * axis[2]],
                        [axis[2] * axis[0], axis[2] * axis[1], axis[2] * axis[2]]])

        Ux = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])

        R = np.identity(3) * np.cos(angle) + (1 - np.cos(angle)) * uuT + Ux * np.sin(angle)

        return R
    
    def rotM2eaa(self, R):
        '''
        Returns the principal axis and angle encoded in the rotation matrix R
        '''

        trace = R[0][0] + R[1][1] + R[2][2]

        angle = np.arccos((trace - 1) / 2)

        if (angle < 1E-8):
      
            axis = np.array([1, 0, 0]) # Axis aleatorio, ya que la matriz identidad hace una rotación de 0 radianes.

        elif (abs(angle) == np.pi):

            uuT = (1/2) * (R + np.identity(3))

            axis = np.array([np.sqrt(uuT[0][0]), np.sqrt(uuT[1][1]), np.sqrt(uuT[2][2])])

        else:
      
            Ux = (1 / (2 * np.sin(angle))) * (R - R.transpose())

            axis = np.array([Ux[2][1], Ux[0][2], Ux[1][0]])
    
        return(axis, angle)
    
    def eAngles2rotM(self, yaw, pitch, roll): #psi, theta, phi
        '''
        Given a set of Euler angles returns the rotation matrix R
        '''

        yaw = yaw * (np.pi/180)
        pitch = pitch * (np.pi/180)
        roll = roll * (np.pi/180)

        if (abs(np.cos(pitch)) < 1E-8):
      
            if (np.sin(pitch) == -1):

                r11 = 0
                r12 = -np.sin(roll + yaw)
                r13 = -np.cos(roll + yaw)

                r21 = 0
                r22 = np.cos(roll + yaw)
                r23 = -np.sin(roll + yaw)

                r31 = 1
                r32 = 0
                r33 = 0

            elif (np.sin(pitch) == 1):

                r11 = 0
                r12 = np.sin(roll - yaw)
                r13 = np.cos(roll - yaw)

                r21 = 0
                r22 = np.cos(roll - yaw)
                r23 = -np.sin(roll - yaw)

                r31 = -1
                r32 = 0
                r33 = 0

        else:

            r11 = np.cos(pitch) * np.cos(yaw)
            r12 = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.cos(roll) * np.sin(yaw)
            r13 = np.cos(yaw) * np.cos(roll) * np.sin(pitch) + np.sin(yaw) * np.sin(roll)

            r21 = np.cos(pitch) * np.sin(yaw)
            r22 = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(roll) * np.cos(yaw)
            r23 = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)

            r31 = -np.sin(pitch)
            r32 = np.cos(pitch) * np.sin(roll)
            r33 = np.cos(pitch) * np.cos(roll)

        R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    
        return R
    
    def rotM2eAngles(self,R): #psi, theta, phi
        '''
        Given a rotation matrix R returns a set of Euler angles 
        '''
        
        if (R[2][0] == 1):

            pitch = np.arcsin(-R[2][0])

            roll = 0

            yaw = np.arctan2(R[0][1], R[0][2])

        elif (R[2][0] == -1):

            pitch = np.arcsin(-R[2][0])

            roll = 0

            yaw = np.arctan2(R[0][1], R[0][2])

        else:

            pitch = np.arcsin(-R[2][0])

            roll = np.arctan2(R[2][1] / np.cos(pitch), R[2][2] / np.cos(pitch))

            yaw = np.arctan2(R[1][0] / np.cos(pitch), R[0][0] / np.cos(pitch))

        
        return (yaw, pitch, roll)
    
    def Eaa2Quat(self, axis, angle):

        qt = np.zeros((4,1))

        qt[0] = np.cos(angle/2)
        qt[1] = np.sin(angle/2) * axis[0]
        qt[2] = np.sin(angle/2) * axis[1]
        qt[3] = np.sin(angle/2) * axis[2]

        return qt
    
    def quat2Eaa(self, qt):
        
        axis = np.zeros(3)

        angle = 2 * np.arccos(qt[0])

        axis[0] = 1/np.sin(angle/2) * qt[1]
        axis[1] = 1/np.sin(angle/2) * qt[2]
        axis[2] = 1/np.sin(angle/2) * qt[3]

        return (axis, angle)
    
    def quat2RotMatrix(self, qt):
            
        qt = qt / np.linalg.norm(qt)

        q0 = qt[0]
        q = qt[1:]

        R = np.zeros((3,3))

        qx = np.array([[0, -q[2][0], q[1][0]],
                       [q[2][0],0, -q[0][0]],
                       [-q[1][0], q[0][0], 0]], dtype = object)

        R = (q0**2 - q.transpose() @ q) * np.identity(3) + 2 * q @ q.transpose() + 2 * q0 * qx

        return R
    
    # Construye un quaternion unitario que representa la rotación
    # de m0 a m1. No es necesario normalizar los vectores de entrada.
    
    def twoVectors2Quat(self, v0, v1):
    
        v = np.cross(v0, v1)

        q = np.array([[1 + np.dot(v0, v1)], [v[0]], [v[1]], [v[2]]])

        q = q / np.linalg.norm(q)

        return q
    
    def mouseCoords2vector(self,x,y,r2):

        mCoords = np.zeros(3)

        if (x**2 + y**2 < (1/2) * r2):

            mCoords[0] = y 
            mCoords[1] = np.sqrt(r2 - (x**2 - y**2))
            mCoords[2] = -x

        elif (x**2 + y**2 >= (1/2) * r2):

            mCoords[0] = y
            mCoords[1] = r2 / (2 * np.sqrt(x**2 + y**2))
            mCoords[2] = -x

        mCoords[0] = mCoords[0] / np.linalg.norm(mCoords)
        mCoords[1] = mCoords[1] / np.linalg.norm(mCoords)
        mCoords[2] = mCoords[2] / np.linalg.norm(mCoords)

        return mCoords
    
    def updateRotM(self, R):      

        self.entry_RotM_11.configure(state = "normal")
        self.entry_RotM_11.delete(0, "end")
        self.entry_RotM_11.insert(0, str(R[0][0]))
        
        self.entry_RotM_12.configure(state = "normal")
        self.entry_RotM_12.delete(0, "end")
        self.entry_RotM_12.insert(0, str(R[0][1]))
        
        self.entry_RotM_13.configure(state = "normal")
        self.entry_RotM_13.delete(0, "end")
        self.entry_RotM_13.insert(0, str(R[0][2]))


        self.entry_RotM_21.configure(state = "normal")
        self.entry_RotM_21.delete(0, "end")
        self.entry_RotM_21.insert(0, str(R[1][0]))
        
        self.entry_RotM_22.configure(state = "normal")
        self.entry_RotM_22.delete(0, "end")
        self.entry_RotM_22.insert(0, str(R[1][1]))

        self.entry_RotM_23.configure(state = "normal")
        self.entry_RotM_23.delete(0, "end")
        self.entry_RotM_23.insert(0, str(R[1][2]))


        self.entry_RotM_31.configure(state = "normal")
        self.entry_RotM_31.delete(0, "end")
        self.entry_RotM_31.insert(0, str(R[2][0]))
        
        self.entry_RotM_32.configure(state = "normal")
        self.entry_RotM_32.delete(0, "end")
        self.entry_RotM_32.insert(0, str(R[2][1]))
        
        self.entry_RotM_33.configure(state = "normal")
        self.entry_RotM_33.delete(0, "end")
        self.entry_RotM_33.insert(0, str(R[2][2]))

    def updateQuat(self, qt):

        self.entry_quat_0.delete(0, "end")
        self.entry_quat_0.insert(0, str(qt[0][0]))

        self.entry_quat_1.delete(0, "end")
        self.entry_quat_1.insert(0, str(qt[1][0]))

        self.entry_quat_2.delete(0, "end")
        self.entry_quat_2.insert(0, str(qt[2][0]))

        self.entry_quat_3.delete(0, "end")
        self.entry_quat_3.insert(0, str(qt[3][0]))

    def updateEaa(self, angle, axis):

        self.entry_AA_angle.delete(0, "end")
        self.entry_AA_angle.insert(0, str(angle))

        self.entry_AA_ax1.delete(0, "end")
        self.entry_AA_ax1.insert(0, str(axis[0]))

        self.entry_AA_ax2.delete(0, "end")
        self.entry_AA_ax2.insert(0, str(axis[1]))

        self.entry_AA_ax3.delete(0, "end")
        self.entry_AA_ax3.insert(0, str(axis[2]))

    def updateRotVector(self, RotV):

        self.entry_rotV_1.delete(0, "end")
        self.entry_rotV_1.insert(0, str(RotV[0]))

        self.entry_rotV_2.delete(0, "end")
        self.entry_rotV_2.insert(0, str(RotV[1]))

        self.entry_rotV_3.delete(0, "end")
        self.entry_rotV_3.insert(0, str(RotV[2]))

    def updateEAngles(self, roll, pitch, yaw):

        self.entry_EA_roll.delete(0, "end")
        self.entry_EA_roll.insert(0,str(roll))

        self.entry_EA_pitch.delete(0, "end")
        self.entry_EA_pitch.insert(0, str(pitch))

        self.entry_EA_yaw.delete(0, "end")
        self.entry_EA_yaw.insert(0, str(yaw))

    def disableRotM(self):

        self.entry_RotM_11.configure(state = "disabled")
        self.entry_RotM_12.configure(state = "disabled")
        self.entry_RotM_13.configure(state = "disabled")

        self.entry_RotM_21.configure(state = "disabled")
        self.entry_RotM_22.configure(state = "disabled")
        self.entry_RotM_23.configure(state = "disabled")
        
        self.entry_RotM_31.configure(state = "disabled")
        self.entry_RotM_32.configure(state = "disabled")
        self.entry_RotM_33.configure(state = "disabled")

    def resetbutton_pressed(self):
        """
        Event triggered function on the event of a push on the button Reset
        """

        self.entry_AA_ax1.delete(0, "end")
        self.entry_AA_ax1.insert(0, "1.0")

        self.entry_AA_ax2.delete(0, "end")
        self.entry_AA_ax2.insert(0, "0.0")

        self.entry_AA_ax3.delete(0, "end")
        self.entry_AA_ax3.insert(0, "0.0")


        self.entry_AA_angle.delete(0, "end")
        self.entry_AA_angle.insert(0, "0.0")
       

        self.entry_rotV_1.delete(0, "end")
        self.entry_rotV_1.insert(0, "0.0")
        
        self.entry_rotV_2.delete(0, "end")
        self.entry_rotV_2.insert(0, "0.0")
        
        self.entry_rotV_3.delete(0, "end")
        self.entry_rotV_3.insert(0, "0.0")
        

        self.entry_EA_roll.delete(0, "end")
        self.entry_EA_roll.insert(0, "0.0")
        
        self.entry_EA_pitch.delete(0, "end")
        self.entry_EA_pitch.insert(0, "0.0")
        
        self.entry_EA_yaw.delete(0, "end")
        self.entry_EA_yaw.insert(0, "0.0")
        

        self.entry_quat_0.delete(0, "end")
        self.entry_quat_0.insert(0, "1.0")
        
        self.entry_quat_1.delete(0, "end")
        self.entry_quat_1.insert(0, "0.0")
        
        self.entry_quat_2.delete(0, "end")
        self.entry_quat_2.insert(0, "0.0")
    
        self.entry_quat_3.delete(0, "end")
        self.entry_quat_3.insert(0, "0.0")

        self.R = np.eye(3)

        self.updateRotM(self.R)
        self.disableRotM()

        self.M = np.array(
            [[-1, -1, 1],   #Nodo 0
            [-1, 1, 1],    #Nodo 1
            [1, 1, 1],      #Nodo 2
            [1, -1, 1],      #Nodo 3
            [-1, -1, -1],    #Nodo 4
            [-1, 1, -1],     #Nodo 5
            [1, 1, -1],     #Nodo 6
            [1, -1, -1]], dtype = float).transpose() #Nodo 7

        self.M = self.R.dot(self.M) #Modifica la matriz de vértices con una matriz de rotación M

        self.update_cube() #Actualiza el cubo

        pass
    
    def apply_AA(self):
        """
        Event triggered function on the event of a push on the button button_AA
        """
        #Example on hot to get values from entries:
        #angle = self.entry_AA_angle.get()
        #Example string to number
        #print(float(angle)*2)
        """"""
        # Obtener valores de ángulo y eje desde las entradas
        angle = float(self.entry_AA_angle.get())
        axis = [float(self.entry_AA_ax1.get()), float(self.entry_AA_ax2.get()), float(self.entry_AA_ax3.get())]

        # Convertir ángulo y eje a matriz de rotación
        rot_matrix = self.Eaa2rotM(angle, axis)

        # Actualizar la matriz de rotación y la visualización
        
        self.R = rot_matrix

        #Actualizar las demás
        #Actualizar Euler Angles
        yaw, pitch, roll = self.rotM2eAngles(self.R)
        self.updateEAngles(roll * (180/np.pi), pitch * (180/np.pi), yaw * (180/np.pi))
        #Actualizar RotVect
        axis_array = np.array(axis)
        rotV = angle * axis_array
        self.updateRotVector(rotV)
        #Actualizar Quat
        q = self.Eaa2Quat(axis,angle)
        self.updateQuat(q)

        self.updateRotM(self.R)

        self.M = self.R.dot(self.M) # Modifica la matriz de vértices con la nueva matriz de rotación
        self.update_cube() #Actualiza el cubo

    def apply_rotV(self):
        """
        Event triggered function on the event of a push on the button button_rotV 
        """
        # Obtener valores de vector de rotación desde las entradas
        rot_vector = [float(self.entry_rotV_1.get()), float(self.entry_rotV_2.get()), float(self.entry_rotV_3.get())]

        # Convertir vector de rotación a matriz de rotación
        rot_matrix = self.rotation_vector_to_matrix(rot_vector)

        # Actualizar la matriz de rotación y la visualización
        self.R = rot_matrix
        self.updateRotM(self.R)

        self.M = self.R.dot(self.M)  # Modifica la matriz de vértices con la nueva matriz de rotación
        self.update_cube()  # Actualiza el cubo
        
        pass

    
    def apply_EA(self):
        """
        Event triggered function on the event of a push on the button button_EA
        """
        pass

    
    def apply_quat(self):
        """
        Event triggered function on the event of a push on the button button_quat
        """
        pass

    
    def onclick(self, event):
        """
        Event triggered function on the event of a mouse click inside the figure canvas
        """
        print("Pressed button", event.button)

        x_fig,y_fig= self.canvas_coordinates_to_figure_coordinates(event.x, event.y) #Extract viewport coordinates

        x = x_fig
        y = y_fig
        r2 = np.sqrt(2)

        self.m0 = self.mouseCoords2vector(x, y, r2)

        print("\nM0:\n")

        print(self.m0)

        if event.button:
            self.pressed = True # Bool to control(activate) a drag (click+move)

        if event.button:
            self.pressed = True # Bool to control(activate) a drag (click+move)


    def onmove(self,event):
        """
        Event triggered function on the event of a mouse motion
        """
        
        #Example
        if self.pressed: #Only triggered if previous click
            x_fig, y_fig = self.canvas_coordinates_to_figure_coordinates(event.x, event.y) #Extract viewport coordinates
            
            #print("x: ", x_fig)
            #print("y", y_fig)
            #print("r2", x_fig * x_fig + y_fig * y_fig)

            #R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

            x = x_fig
            y = y_fig
            r2 = np.sqrt(2)
            
            self.m1 = self.mouseCoords2vector(x, y, r2)

            print("\nM1:\n")

            print(self.m1)

            self.δQk = self.twoVectors2Quat(self.m0, self.m1)

            print("\nδQk:\n")

            print(self.δQk)

            self.updateQuat(self.δQk)

            self.R = self.quat2RotMatrix(self.δQk)

            print(self.R)
            
            axis, angle = self.rotM2eaa(self.R)

            self.updateEaa(angle * (180/np.pi), axis)

            rotV = angle * axis

            self.updateRotVector(rotV)

            yaw, pitch, roll = self.rotM2eAngles(self.R)

            self.updateEAngles(roll * (180/np.pi), pitch * (180/np.pi), yaw * (180/np.pi))

            self.m0 = self.m1

            self.updateRotM(self.R)
            
            self.disableRotM()
                    
            self.M = self.R.dot(self.M) #Modifica la matriz de vértices con una matriz de rotación M

            self.update_cube() #Actualiza el cubo


    def onrelease(self,event):
        """
        Event triggered function on the event of a mouse release
        """
        self.pressed = False # Bool to control(deactivate) a drag (click+move)


    def init_cube(self):
        """
        Initialization function that sets up cube's geometry and plot information
        """

        self.M = np.array(
            [[ -1,  -1, 1],   #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose() #Node 7

        self.con = [
            [0, 1, 2, 3], #Face 1
            [4, 5, 6, 7], #Face 2
            [3, 2, 6, 7], #Face 3
            [0, 1, 5, 4], #Face 4
            [0, 3, 7, 4], #Face 5
            [1, 2, 6, 5]] #Face 6

        faces = []

        for row in self.con:
            faces.append([self.M[:,row[0]],self.M[:,row[1]],self.M[:,row[2]],self.M[:,row[3]]])

        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection='3d')

        for item in [self.fig, ax]:
            item.patch.set_visible(False)

        self.facesObj = Poly3DCollection(faces, linewidths=.2, edgecolors='k',animated = True)
        self.facesObj.set_facecolor([(0,0,1,0.9), #Blue
        (0,1,0,0.9), #Green
        (.9,.5,0.13,0.9), #Orange
        (1,0,0,0.9), #Red
        (1,1,0,0.9), #Yellow
        (0,0,0,0.9)]) #Black

        #Transfering information to the plot
        ax.add_collection3d(self.facesObj)

        #Configuring the plot aspect
        ax.azim=-90
        ax.roll = -90
        ax.elev=0   
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        ax.set_zlim3d(-2, 2)
        ax.set_aspect('equal')
        ax.disable_mouse_rotation()
        ax.set_axis_off()

        self.pix2unit = 1.0/60 #ratio for drawing the cube 


    def update_cube(self):
        """
        Updates the cube vertices and updates the figure.
        Call this function after modifying the vertex matrix in self.M to redraw the cube
        """

        faces = []

        for row in self.con:
            faces.append([self.M[:,row[0]],self.M[:,row[1]],self.M[:,row[2]], self.M[:,row[3]]])

        self.facesObj.set_verts(faces)
        self.bm.update()


    def canvas_coordinates_to_figure_coordinates(self,x_can,y_can):
        """
        Remap canvas coordinates to cube centered coordinates
        """

        (canvas_width,canvas_height)=self.canvas.get_width_height()
        figure_center_x = canvas_width/2+14
        figure_center_y = canvas_height/2+2
        x_fig = (x_can-figure_center_x)*self.pix2unit
        y_fig = (y_can-figure_center_y)*self.pix2unit

        return(x_fig,y_fig)


    def destroy(self):
        """
        Close function to properly destroy the window and tk with figure
        """
        try:
            self.destroy()
        finally:
            exit()


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
            cv.draw_idle()


if __name__ == "__main__":
    app = Arcball()
    app.mainloop()
    exit()
