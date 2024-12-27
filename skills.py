import os
import webbrowser
import sys
import subprocess
import os

def browser():
    '''Открывает браузер заданнный по уполчанию в системе с url указанным здесь'''

    webbrowser.open('https://www.youtube.com', new=2)



def offpc():
    # Эта команда отключает ПК под управлением Windows

    # os.system('shutdown \s')
    print('пк был бы выключен, но команде # в коде мешает;)))')



def offBot():
    '''Отключает бота'''
    sys.exit()


def passive():
    '''Функция заглушка при простом диалоге с ботом'''
    pass

def run_yandexmusic():
    os.startfile('yndmusic.exe.lnk')
def run_cmd():
    os.startfile('cmd.exe.lnk')
def run_discord():
    os.startfile('Discord_run.lnk')
