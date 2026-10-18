; Installer for TDPlay AI Video Render on Windows (Inno Setup 6).
; Build:  iscc /DAppVersion=1.2.0 windows\installer\TDPlay.iss   ->  dist\TDPlay-AI-Video-Render-Setup-1.2.0.exe
;
; It installs the app for the current user (no administrator rights), adds it to the Start menu, and
; then runs windows\install.ps1, which finds the graphics card and installs Python, PyTorch for that
; card (AMD ROCm or NVIDIA CUDA), Lightricks' LTX-2 and ffmpeg next to the app. Those are several GB,
; so they are downloaded during the installation instead of being inside this file.
; "/setupargs=..." on the command line is passed on to install.ps1 (the tests use "-TestCpu -SkipModel").

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName "TDPlay AI Video Render"
#define Root "..\.."

[Setup]
AppId={{E14E6DD1-72E2-4F6A-B1C3-E7A7CFCF2D5F}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=TDPlay
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#Root}\dist
OutputBaseFilename=TDPlay-AI-Video-Render-Setup-{#AppVersion}
SetupIconFile={#Root}\tdplay\assets\AppIcon.ico
UninstallDisplayIcon={app}\tdplay\assets\AppIcon.ico
UninstallDisplayName={#AppName}
InfoBeforeFile=before-install.txt
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19041

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#Root}\tdplay\*"; DestDir: "{app}\tdplay"; Excludes: "__pycache__,*.pyc"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Root}\windows\install.ps1"; DestDir: "{app}\windows"; Flags: ignoreversion
Source: "{#Root}\windows\requirements-windows.txt"; DestDir: "{app}\windows"; Flags: ignoreversion
Source: "{#Root}\start.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m tdplay.desktop_app"; WorkingDir: "{app}"; IconFilename: "{app}\tdplay\assets\AppIcon.ico"; AppUserModelID: "TDPlay.AIVideoRender"; Comment: "One continuous AI video shot, made on your PC"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m tdplay.desktop_app"; WorkingDir: "{app}"; IconFilename: "{app}\tdplay\assets\AppIcon.ico"; AppUserModelID: "TDPlay.AIVideoRender"; Comment: "One continuous AI video shot, made on your PC"; Tasks: desktopicon
Name: "{autoprograms}\{#AppName} - repair setup"; Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\windows\install.ps1"" -FromInstaller"; WorkingDir: "{app}"; IconFilename: "{app}\tdplay\assets\AppIcon.ico"; Comment: "Install the app's Python packages again"

[Run]
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m tdplay.desktop_app"; WorkingDir: "{app}"; Description: "Open {#AppName}"; Flags: postinstall nowait skipifsilent; Check: SetupSucceeded

[UninstallDelete]
; Made by install.ps1, not by this installer: Python, PyTorch, LTX-2, ffmpeg.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\vendor"
Type: filesandordirs; Name: "{app}\tdplay"

[Code]
var
  SetupOk: Boolean;

function SetupSucceeded: Boolean;
begin
  Result := SetupOk;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Args: String;
begin
  if CurStep <> ssPostInstall then
    Exit;
  Args := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\windows\install.ps1') +
          '" -FromInstaller ' + ExpandConstant('{param:setupargs|}');
  if WizardSilent then
    Args := Args + ' -SkipModel -NoPause';
  WizardForm.StatusLabel.Caption := 'Setting up Python, PyTorch for your graphics card and LTX-2. ' +
    'This downloads a few GB; the window that opened shows the progress.';
  WizardForm.ProgressGauge.Style := npbstMarquee;
  Log('Running install.ps1 ' + Args);
  SetupOk := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Args, ExpandConstant('{app}'),
                  SW_SHOW, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  Log('install.ps1 exit code: ' + IntToStr(ResultCode));
  WizardForm.ProgressGauge.Style := npbstNormal;
  if not SetupOk then
    SuppressibleMsgBox('The setup did not finish (exit code ' + IntToStr(ResultCode) + ').' + #13#10#13#10 +
      'Fix what the setup window said, then open "{#AppName} - repair setup" from the Start menu.',
      mbError, MB_OK, IDOK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;
  Data := ExpandConstant('{%USERPROFILE}\TDPlay');
  if DirExists(Data) and (SuppressibleMsgBox('Also delete the downloaded models and your videos?' + #13#10#13#10 +
      Data, mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES) then
    DelTree(Data, True, True, True);
end;
