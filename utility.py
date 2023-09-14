import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

class Env:
    def __init__(self, width, height,margin_horizontal,margin_vertical, numVerticalLines,numHorizontalLines):
        self.width=width #600
        self.height=height #200
        self.img = np.ones(shape=(self.height, self.width, 3)) * 255.0
        self.margin_horizontal = margin_horizontal #6
        self.margin_vertical=margin_vertical #2
        self.numVerticalLines=numVerticalLines #13
        self.numHorizontalLines= numHorizontalLines #5



    # Creates cliff walking grid
    def initialize_frame(self):
            # Vertical Lines
            for i in range(self.numVerticalLines):
                self.img = cv2.line(self.img, (49 * i + self.margin_horizontal, self.margin_vertical),
                               (49 * i + self.margin_horizontal, 200 - self.margin_vertical), color=(0, 0, 0), thickness=1)
            # Horizontal Lines
            for i in range(self.numHorizontalLines):
                self.img = cv2.line(self.img, (self.margin_horizontal, 49 * i + self.margin_vertical),
                               (600 - self.margin_horizontal, 49 * i + self.margin_vertical), color=(0, 0, 0), thickness=1)

            # Cliff Box
            self.img = cv2.rectangle(self.img, (49 * 1 + self.margin_horizontal + 2, 49 * 3 + self.margin_vertical + 2),
                                (49 * 11 + self.margin_horizontal - 2, 49 * 4 + self.margin_vertical - 2), color=(255, 0, 255),
                                thickness=-1)
            self.img = cv2.putText(self.img, text="Cliff", org=(49 * 5 + self.margin_horizontal, 49 * 4 + self.margin_vertical - 10),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # Goal
            self.img = cv2.putText(self.img, text="G", org=(49 * 11 +self. margin_horizontal + 10, 49 * 4 + self.margin_vertical - 10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

            return self.img

    def plot_policy(self, learnedPolicy):

        pilimg = Image.fromarray((self.img).astype(np.uint8))
        draw = ImageDraw.Draw(pilimg)
        font = ImageFont.truetype("Arial.ttf", 16)
        for state in range(len(learnedPolicy) - 1):
            if state < 37 or state > 46:
                # print(state)
                row, column = np.unravel_index(indices=state, shape=(4, 12))

                if learnedPolicy[state] == 0:
                    arrow = "\u2191"
                    draw.text((49 * column + self.margin_horizontal + 10, 49 * row + self.margin_vertical + 10), arrow,
                              (0, 0, 0), font=font)

                elif learnedPolicy[state] == 1:
                    arrow = "\u2192"
                    draw.text((49 * column + self.margin_horizontal + 10, 49 * row + self.margin_vertical + 10), arrow,
                              (0, 0, 0), font=font)

                elif learnedPolicy[state] == 2:
                    arrow = "\u2193"
                    draw.text((49 * column + self.margin_horizontal + 10, 49 * row + self.margin_vertical + 10), arrow,
                              (0, 0, 0), font=font)

                elif learnedPolicy[state] == 3:
                    arrow = "\u2190"
                    draw.text((49 * column + self.margin_horizontal + 10, 49 * row + self.margin_vertical + 10), arrow,
                              (0, 0, 0), font=font)
        self.img = np.array(pilimg)
        return self.img

def generate_heatmap(Qtable):
        """
            Generates heatmap to visualize agent's learned actions on the environment
        """
        import seaborn as sns;
        sns.set()
        # display mean of environment values using a heatmap
        data = np.mean(Qtable, axis=1)
        data = data.reshape((4, 12))
        ax=sns.heatmap(np.array(data),annot=True,annot_kws={'size': 10},fmt=".1f",linewidth=.5)
        plt.show()
        # figure = ax.get_figure()
        # figure.savefig('sarsa-heatmap.png', dpi=400)
        return ax

def state_array(numberStates, state):

    # Initialze current state array, e.g., [0. 0. 0. 0. 0. 0. 0. 0. 0.]
    state_arr = np.zeros(numberStates)
    # Set the agent position in the current state array, i.e., [[1. 0. 0. 0. 0. 0. 0. 0. 0.]]
    state_arr[state] = 1
    # Reshape array
    state_arr = np.reshape(state_arr, [1, numberStates])
    return state_arr

def save_data(data,agent):
    array_to_save = np.array(data)
    # print("memory to save",array_to_save)
    with open("saved_episodes"+agent+".csv", "a") as outfile:
        np.savetxt(outfile, array_to_save,delimiter=',', fmt='%s')
