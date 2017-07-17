import threading
import visualizations
from threading import Thread
import os

def func1():
    visualizations.scatterKillInjured('data/GTD_FULL_UNKNOWNATTACKS_FILTERED.csv', title="Unknown Attacks")

def func2():
    visualizations.scatterKillInjured('data/GTD_FULL_KNOWNATTACKS_FILTERED.csv', title="Known Attacks")

if __name__ == '__main__':
	Thread(target = func1).start()
	Thread(target = func2).start()