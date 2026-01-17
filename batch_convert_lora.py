import os
import subprocess
import glob
import sys

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.widgets import Frame
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window


def get_convertible_files(output_dir):
    """获取可转换的文件列表（排除已转换的）"""
    all_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    convertible = []

    for file_path in all_files:
        filename = os.path.basename(file_path)

        if "_changed" in filename:
            continue

        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_changed{ext}")
        if os.path.exists(output_path):
            continue

        convertible.append(file_path)

    return sorted(convertible)


def convert_file(file_path, output_dir):
    """转换单个文件"""
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_changed{ext}")

    cmd = [
        sys.executable, "src/musubi_tuner/convert_lora.py",
        "--input", file_path,
        "--output", output_path,
        "--target", "other"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"完成: {filename} -> {os.path.basename(output_path)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {filename}, 错误: {e}")
        return False


def select_files(files):
    """交互式选择文件"""
    if not files:
        return []

    current_index = 0
    selected = set()
    result = None

    kb = KeyBindings()

    @kb.add('up')
    @kb.add('k')
    def move_up(event):
        nonlocal current_index
        current_index = (current_index - 1) % len(files)

    @kb.add('down')
    @kb.add('j')
    def move_down(event):
        nonlocal current_index
        current_index = (current_index + 1) % len(files)

    @kb.add('space')
    def toggle_select(event):
        if current_index in selected:
            selected.remove(current_index)
        else:
            selected.add(current_index)

    @kb.add('a')
    def select_all(event):
        nonlocal selected
        if len(selected) == len(files):
            selected.clear()
        else:
            selected = set(range(len(files)))

    @kb.add('enter')
    def confirm(event):
        nonlocal result
        result = [files[i] for i in sorted(selected)]
        event.app.exit()

    @kb.add('q')
    @kb.add('escape')
    def quit(event):
        nonlocal result
        result = []
        event.app.exit()

    def get_formatted_text():
        lines = [('class:title', '选择要转换的文件:\n')]
        lines.append(('', '-' * 50 + '\n'))

        for i, f in enumerate(files):
            filename = os.path.basename(f)
            prefix = '>' if i == current_index else ' '
            checkbox = '[x]' if i in selected else '[ ]'

            if i == current_index:
                lines.append(('class:selected', f' {prefix} {checkbox} {filename}\n'))
            else:
                lines.append(('', f' {prefix} {checkbox} {filename}\n'))

        lines.append(('', '-' * 50 + '\n'))
        lines.append(('class:hint', ' ↑/↓:移动  空格:选择  a:全选  回车:确认  q:退出\n'))

        return FormattedText(lines)

    content = Window(
        content=FormattedTextControl(get_formatted_text),
        always_hide_cursor=True,
    )

    layout = Layout(content)

    style_dict = {
        'selected': 'reverse',
        'title': 'bold',
        'hint': 'italic',
    }

    from prompt_toolkit.styles import Style
    style = Style.from_dict(style_dict)

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
        refresh_interval=0.1,
    )

    app.run()
    return result


def main():
    output_dir = "output"

    files = get_convertible_files(output_dir)

    if not files:
        print("没有需要转换的文件")
        return

    selected = select_files(files)

    if not selected:
        print("已取消")
        return

    print(f"\n将转换 {len(selected)} 个文件:\n")

    for file_path in selected:
        print(f"转换中: {os.path.basename(file_path)}")
        convert_file(file_path, output_dir)

    print("\n转换完成!")


if __name__ == "__main__":
    main()
