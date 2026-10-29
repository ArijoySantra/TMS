; Inno Setup script: turns dist\TrafficAI into one TrafficAI_Setup.exe.
; Build_TrafficAI_EXE.bat compiles it automatically when Inno Setup 6 is installed.
#define AppName "Traffic AI"
#define AppVersion GetEnv("TRAFFIC_AI_VERSION")
#if AppVersion == ""
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{5B1E6C2A-7F3D-4C1B-9E1A-2D7C3A9F4B10}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Traffic AI project
DefaultDirName={localappdata}\Programs\TrafficAI
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Installs for the current user only: no administrator rights needed, and the app can write its results.
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=TrafficAI_Setup
SetupIconFile=TrafficAI.ico
UninstallDisplayIcon={app}\TrafficAI.exe
Compression=lzma2/fast
SolidCompression=no
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\TrafficAI\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\TrafficAI.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\TrafficAI.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\TrafficAI.exe"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent
