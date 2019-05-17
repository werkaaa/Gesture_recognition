import webbrowser
import numpy as np

def show_cats(arr):
    global catsshown
    if(np.prod(arr)==1 and not catsshown):
        webbrowser.open('https://imgflip.com/i/2jlwbi', new=2)
        catsshown = True


if __name__=="__main__":
    arr = [1, 1, 1]
    catsshown = False
    for i in range(5):
        show_cats(arr)

