
import curses
import csv
import abc
from PIL import Image
from pathlib import Path
from datetime import datetime
from jetbot import Robot, Camera

SCRIPT_DIR = Path(__file__).resolve().parent

class Agent:
    max_speed = 1
    min_speed = -1
    fields = ("image", "action")
    
    def __init__(self):
        self.robot = Robot()
        self.camera = Camera.instance(width=224, height=224)
        self.left_speed = 0
        self.right_speed = 0
        
        self.save_dir = SCRIPT_DIR / "plays" / f"{datetime.now():%m%d%H%M}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_log = self.save_dir  / "log.csv"
        self.frames = 0
        
        with open(self.save_log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    @abc.abstractmethod
    def step(self, action):
        pass

    def _accelerate_left_wheel(self, speed: float = 0.1):
        self.left_speed += speed
        if self.left_speed > self.max_speed:
            self.left_speed = self.max_speed
        elif self.left_speed < self.min_speed:
            self.left_speed = self.min_speed
                
    def _accelerate_right_wheel(self, speed: float = 0.1):
        self.right_speed += speed
        if self.right_speed > self.max_speed:
            self.right_speed = self.max_speed
        elif self.right_speed < self.min_speed:
            self.right_speed = self.min_speed
            
    def _play(self, stdscr):
        curses.curs_set(0)  # Hide the cursor
        stdscr.nodelay(1)  # Don't wait for input when calling getch
        stdscr.timeout(100)  # Wait for 100ms for a key press

        help_lines = [
            "Control the robot with",
            " ----- ",
            "'w' or '↑' to move forward",
            "'a' or '←' to turn left",
            "'s' or '↓' to move backward",
            "'d' or '→' to turn right",
            "'p' to stop the car",
            "'q' or 'esc' to exit the program",
        ]
        help_text = "\n".join(help_lines)
        stdscr.addstr(0, 0, help_text)

        while True:
            key = stdscr.getch()

            if key == ord('q') or key == curses.KEY_EXIT:
                stdscr.addstr(len(help_lines), 0, "Pressed q, exiting")
                raise KeyboardInterrupt

            if key == ord('w') or key == curses.KEY_UP:
                stdscr.addstr(len(help_lines), 0, "Pressed w         ")
                self.step(0)
            elif key == ord('d') or key == curses.KEY_RIGHT:
                stdscr.addstr(len(help_lines), 0, "Pressed d         ")
                self.step(1)
            elif key == ord('a') or key == curses.KEY_LEFT:
                stdscr.addstr(len(help_lines), 0, "Pressed a         ")
                self.step(2)
            elif key == ord('s') or key == curses.KEY_DOWN:
                stdscr.addstr(len(help_lines), 0, "Pressed s         ")
                self.step(3)
            elif key == ord('p'):
                stdscr.addstr(len(help_lines), 0, "Pressed p         ")
                self.step(4)
            else:
                stdscr.addstr(len(help_lines), 0, "Nothing pressed   ")
                self.step(5)

    def play(self):
        curses.wrapper(self._play)
        
    def stop_playing(self):
        self.robot.set_motors(0, 0)
        self.camera.stop()

class RacerAgent(Agent):
    def step(self, action):
        if action == 0:
            self._accelerate_left_wheel()
            self._accelerate_right_wheel()
        elif action == 3:
            self._accelerate_left_wheel(-0.1)
            self._accelerate_right_wheel(-0.1)
            
        l, r = self.left_speed, self.right_speed

        if action == 1:
            r /= 2
        elif action == 2:
            l /= 2
        elif action == 4:
            l, r = 0, 0
            
        self.robot.set_motors(l, r)
        
        self.frames += 1
        if action == 5 and not self.frames % 4 == 0:
            return

        image = self.camera.value
        image_path = f"{self.save_dir}/{self.frames:05d}.png"

        img = Image.fromarray(image)
        img.save(image_path)
        with open(self.save_log, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({ "image": image_path.split("/")[-1], "action": action })


def main() -> None:
    try:
        agent = RacerAgent()
        agent.play()
    except KeyboardInterrupt:
        agent.stop_playing()
        print("Exiting program and stopping the robot.")


if __name__ == '__main__':
    main()

