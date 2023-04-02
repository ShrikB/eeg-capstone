import PySimpleGUI as sg


#GUI Stuff
layout = [
    [sg.Text('Enter Test Name:'), sg.InputText()],
    [sg.Button('Analyze')],
    [sg.Text('Open Previous Result'), sg.Input(key='-IN-'), sg.FileBrowse(file_types=(("Excel Files", "*.xls"),))],
    [sg.Button('Open Result')],
    [sg.Button('Exit App')]
]

window = sg.Window('Sleep Analysis', layout)

while True:
    event, values = window.read()
    if event == 'Analyze':
        name = values[0]    #placeholder
        sg.popup(f'Measuring')
    if event == 'Open Result':
        sg.popup(f'Running')    #placeholder
    if event in (sg.WIN_CLOSED, "Exit App"):
        break

window.close()
