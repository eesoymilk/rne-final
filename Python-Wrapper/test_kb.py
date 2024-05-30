import msvcrt as getch


def main() -> None:
    while True:
        key = getch.getch().decode('utf-8')
        print(key)

        if key == 'q':
            break


if __name__ == '__main__':
    main()
