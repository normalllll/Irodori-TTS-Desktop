/* Irodori-TTS-Desktop.exe: starts the package's own Python with the desktop app and exits.

   The package keeps app, runtime and resources next to this file. Arguments are passed on
   to desktop_app.py (for example --data-dir). Built by scripts/build_desktop.py with the
   C runtime linked in, so it needs nothing installed. */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wchar.h>

#define PATH_CHARS 4096
#define TITLE L"Irodori TTS Desktop"

static void fail(const wchar_t *message)
{
    MessageBoxW(NULL, message, TITLE, MB_ICONERROR | MB_OK);
}

/* The command line after the program's own name. */
static const wchar_t *arguments(void)
{
    const wchar_t *p = GetCommandLineW();
    if (*p == L'"') {
        for (p++; *p && *p != L'"'; p++) {
        }
        if (*p) {
            p++;
        }
    } else {
        while (*p && *p != L' ' && *p != L'\t') {
            p++;
        }
    }
    while (*p == L' ' || *p == L'\t') {
        p++;
    }
    return p;
}

static void set_path(const wchar_t *name, const wchar_t *root, const wchar_t *relative)
{
    wchar_t value[PATH_CHARS];
    swprintf(value, PATH_CHARS, L"%s\\%s", root, relative);
    SetEnvironmentVariableW(name, value);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR line, int show)
{
    (void)instance;
    (void)previous;
    (void)line;
    (void)show;
    wchar_t root[PATH_CHARS], python[PATH_CHARS], app[PATH_CHARS];
    DWORD length = GetModuleFileNameW(NULL, root, PATH_CHARS);
    wchar_t *slash = length && length < PATH_CHARS ? wcsrchr(root, L'\\') : NULL;
    if (!slash) {
        fail(L"无法确定程序所在的文件夹。\n\nCould not find the folder this program is in.");
        return 1;
    }
    *slash = L'\0';
    swprintf(python, PATH_CHARS, L"%s\\runtime\\python\\pythonw.exe", root);
    swprintf(app, PATH_CHARS, L"%s\\app", root);
    if (GetFileAttributesW(python) == INVALID_FILE_ATTRIBUTES) {
        fail(L"找不到 runtime 文件夹。请解压整个文件夹，不要只复制这个程序。\n\n"
             L"The runtime folder is missing. Extract the whole folder, not only this program.");
        return 1;
    }

    /* Downloads and add-ons stay in the package; a Python set up elsewhere on the computer
       must not leak in. */
    set_path(L"IRODORI_CACHE", root, L"resources");
    set_path(L"HF_HOME", root, L"resources\\huggingface");
    set_path(L"TORCH_HOME", root, L"resources\\torch");
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    SetEnvironmentVariableW(L"PYTHONIOENCODING", L"utf-8");
    SetEnvironmentVariableW(L"PYTHONNOUSERSITE", L"1");
    SetEnvironmentVariableW(L"PYTHONHOME", NULL);
    SetEnvironmentVariableW(L"PYTHONPATH", NULL);

    const wchar_t *rest = arguments();
    size_t size = wcslen(python) + wcslen(app) + wcslen(rest) + 64;
    wchar_t *command = HeapAlloc(GetProcessHeap(), 0, size * sizeof(wchar_t));
    if (!command) {
        fail(L"内存不足。\n\nOut of memory.");
        return 1;
    }
    swprintf(command, size, L"\"%s\" \"%s\\desktop_app.py\" %s", python, app, rest);

    STARTUPINFOW startup = {sizeof(startup)};
    PROCESS_INFORMATION process;
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, 0, NULL, app, &startup, &process)) {
        DWORD error = GetLastError();
        wchar_t message[PATH_CHARS];
        swprintf(message, PATH_CHARS,
                 L"无法启动程序（错误 %lu）。\n\nCould not start the application (error %lu).",
                 error, error);
        fail(message);
        return 1;
    }
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return 0;
}
