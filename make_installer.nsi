; Build Unicode installer
Unicode True
!include "LogicLib.nsh"
!include "StrFunc.nsh"
${StrLoc}

!define VERSION "1.0.5" ; Se incrementa automáticamente por bump2version

LoadLanguageFile "${NSISDIR}\Contrib\Language files\Spanish.nlf"

; Nombre del instalador

Name "AntTracker v${VERSION}"

; Nombre de archivo del instalador
OutFile "AntTracker_v${VERSION}_installer.exe"

; Request application privileges for Windows Vista and higher
RequestExecutionLevel admin

; The default installation directory
InstallDir $PROGRAMFILES64\AntTracker

; Registry key to check for directory (so if you install again, it will
; overwrite the old one automatically)
InstallDirRegKey HKLM "Software\AntTracker" "Install_Dir"

;--------------------------------

; Pages

Page license
LicenseData "LICENSE"
Page components
Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

!define PathUnquoteSpaces '!insertmacro PathUnquoteSpaces '
Function PathUnquoteSpaces
Exch $0
Push $1
StrCpy $1 $0 1
StrCmp $1 '"' 0 ret
StrCpy $1 $0 "" -1
StrCmp $1 '"' 0 ret
StrCpy $0 $0 -1 1
ret:
Pop $1
Exch $0
FunctionEnd
!macro PathUnquoteSpaces var
Push ${var}
Call PathUnquoteSpaces
Pop ${var}
!macroend

;--------------------------------
; The "" makes the section hidden.
Section "" SecUninstallPrevious
  SectionIn RO
  ; Check for uninstaller.
  ReadRegStr $R0 HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\AntTracker" "Install_Dir"
  ${If} $R0 == ""
      Goto Done
  ${EndIf}
  ${PathUnquoteSpaces} $R0
  RMDir /r $R0
  Done:
SectionEnd
; The stuff to install
Section "AntTracker & AntLabeler (requerido)"
  SectionIn RO

  ; Agregar forzosamente la carpeta \AntTracker al directorio
  ; de instalación si el usuario no lo especifica
  ${StrLoc} $0 $INSTDIR "\AntTracker" ">"
  ${If} $0 == ""
    ${StrLoc} $0 "$INSTDIR" "\" "<"
    ${If} $0 != 0
      StrCpy $INSTDIR "$INSTDIR\AntTracker"
    ${Else}
      StrCpy $INSTDIR "$INSTDIRAntTracker"
    ${EndIf}
  ${EndIf}
  SetOutPath $INSTDIR

  ; Put files there
  File /r "dist\AntTracker\*"

  ; Write the installation path into the registry
  WriteRegStr HKLM "Software\AntTracker" "Install_Dir" "$INSTDIR"

  ; Write the uninstall keys for Windows
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\AntTracker" "DisplayName" "AntTracker"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\AntTracker" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\AntTracker" "Install_Dir" '"$INSTDIR"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\AntTracker" "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\AntTracker" "NoRepair" 1
  WriteUninstaller "$INSTDIR\uninstall.exe"

SectionEnd

Section "Visual C++ Redistributable para Visual Studio 2017"
  SectionIn RO
  NSISdl::download https://aka.ms/vs/15/release/vc_redist.x64.exe "$INSTDIR\vc_redist.x64.exe"
  ExecWait '"$INSTDIR\vc_redist.x64.exe" /install /passive /norestart'
SectionEnd

Section "Accesos directos (Menu Inicio)"
  CreateDirectory "$SMPROGRAMS\AntTracker"
  CreateShortcut "$SMPROGRAMS\AntTracker\Uninstall.lnk" "$INSTDIR\uninstall.exe"
  CreateShortcut "$SMPROGRAMS\AntTracker\AntTracker.lnk" "$INSTDIR\AntTracker.exe"
  CreateShortcut "$SMPROGRAMS\AntTracker\AntLabeler.lnk" "$INSTDIR\AntLabeler.exe"
SectionEnd

Section /o "Accesos directos (Escritorio)"
  CreateShortcut "$DESKTOP\AntTracker.lnk" "$INSTDIR\AntTracker.exe"
  CreateShortcut "$DESKTOP\AntLabeler.lnk" "$INSTDIR\AntLabeler.exe"
SectionEnd

;--------------------------------

; Uninstaller

Section "Uninstall"
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\AntTracker"
  DeleteRegKey HKLM "Software\AntTracker"

  ; Remove files and uninstaller
  Delete "$INSTDIR\*"

  ; Remove shortcuts, if any
  Delete "$SMPROGRAMS\AntTracker\*.lnk"
  Delete "$DESKTOP\AntTracker.lnk"
  Delete "$DESKTOP\AntLabeler.lnk"

  ; Remove directories
  RMDir "$SMPROGRAMS\AntTracker"
  RMDir /r "$INSTDIR"
SectionEnd
