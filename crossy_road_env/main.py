import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

CAR_COLORS = [(255, 0, 0)]

class CrossyRoadEnv(gym.Env):
    
    def __init__(self):
        super().__init__()
        
        self.width = 40
        self.height = 22
        self.observation_space = spaces.Box(low=0, high=5, shape=(self.height, self.width), dtype=np.int32)
        self.action_space = spaces.Discrete(5)  # 0: up, 1: down, 2: left, 3: right
        
        # Inicjalizacja PyGame do renderowania
        pygame.init()
        self.screen = pygame.display.set_mode((self.width * 40, self.height * 40))
        self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.agent_pos = [self.width // 2, self.height - 1]
        self.cars = []
        self.logs = []
        self.water_rows = set()
        self.car_rows = set()
        self.done = False
        self.score = 0

        reserved_rows = [self.height - 0, self.height - 1, self.height - 2, self.height - 3]  # Zarezerwowane wiersze
        all_valid_rows = [y for y in range(self.height - 1) if y not in reserved_rows]

        # Losuj rzędy z wodą (kłodami)
        self.water_rows = set(random.sample(all_valid_rows, 3))

        # Losuj rzędy z autami (bez wody i bez reserved)
        possible_car_rows = list(set(all_valid_rows) - self.water_rows)
        self.car_rows = set(random.sample(possible_car_rows, 5))

        # Twórz auta – po 1 na rząd
        for row in self.car_rows:
            x = random.randint(0, self.width - 1)
            direction = random.choice([-1, 1])
            color = random.choice(CAR_COLORS)
            self.cars.append([x, row, direction, color])

        # Twórz kłody – po 1–2 na rząd (długość 3)
        for row in self.water_rows:
            direction = random.choice([-1, 1])
            for _ in range(random.randint(1, 2)):
                start_x = random.randint(0, self.width - 3)
                self.logs.append([start_x, row, direction])
                
        # Dodaj skały – np. 20 losowych
        self.rocks = []
        for _ in range(10):
            attempt = 0
            while attempt < 100:  # nie nieskończoność
                rx = random.randint(0, self.width - 1)
                ry = random.randint(0, self.height - 1)

                if (
                    ry in self.water_rows or
                    ry in self.car_rows or
                    ry >= self.height - 4 or
                    [rx, ry] == self.agent_pos or
                    [rx, ry] in self.rocks
                ):
                    attempt += 1
                    continue
                
                self.rocks.append([rx, ry])
                break


        return self._get_obs(), {}


    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}
        
        # Ruch agenta
        x, y = self.agent_pos
        if action == 0 and y > 0:  # Góra
            y -= 1
        elif action == 1 and y < self.height - 1:  # Dół
            y += 1
        elif action == 2 and x > 0:  # Lewo
            x -= 1
        elif action == 3 and x < self.width - 1:  # Prawo
            x += 1
        elif action == 4:  # Pozostań w miejscu
            pass
        # self.agent_pos = [x, y]

        if [x, y] not in self.rocks:
            self.agent_pos = [x, y]
        # jeśli skała — nie zmieniaj pozycji (agent zostaje w miejscu)

        
        # Ruch samochodów i kłód
        new_cars = []
        for car in self.cars:
            car[0] += car[2]

            # Auto zniknęło z planszy – generuj nowe z drugiej strony
            if car[2] == 1 and car[0] > self.width:
                new_x = -1
                new_car = [new_x, car[1], car[2], random.choice(CAR_COLORS)]
                new_cars.append(new_car)
            elif car[2] == -1 and car[0] < -1:
                new_x = self.width
                new_car = [new_x, car[1], car[2], random.choice(CAR_COLORS)]
                new_cars.append(new_car)
            else:
                new_cars.append(car)

        self.cars = new_cars

        
        # for log in self.logs:
        #     log[0] += log[2]
        #     if log[0] < 0 or log[0] >= self.width:
        #         log[2] *= -1

        new_logs = []
        for log in self.logs:
            log[0] += log[2]  # przesuwaj

            # Jeśli cała kłoda wyszła poza planszę – usuń i dodaj nową z drugiej strony
            if log[2] == 1 and log[0] > self.width:
                # z lewej strony nowa
                new_start = -3
                new_logs.append([new_start, log[1], log[2]])
            elif log[2] == -1 and log[0] < -3:
                # z prawej strony nowa
                new_start = self.width
                new_logs.append([new_start, log[1], log[2]])
            else:
                new_logs.append(log)
        self.logs = new_logs

        if self.agent_pos[1] in self.water_rows:
            for log in self.logs:
                if log[1] == self.agent_pos[1] and log[0] <= self.agent_pos[0] < log[0] + 3:
                    self.agent_pos[0] += log[2]
                    break  # wystarczy jedna kłoda
        # Ograniczenia planszy po przesunięciu z kłodą
        self.agent_pos[0] = max(0, min(self.agent_pos[0], self.width - 1))
        
        # Sprawdź kolizje
        reward = -0.1  # Mała kara za każdy krok
        terminated = False
        
        # Kolizja z samochodem
        for car in self.cars:
            if self.agent_pos[0] == car[0] and self.agent_pos[1] == car[1]:
                reward = -10
                terminated = True
        
        # Kolizja z wodą (jeśli nie na kłodzie)
        if self.agent_pos[1] in self.water_rows:# Obszar rzeki
            on_log = False
            for log in self.logs:
                if log[1] == self.agent_pos[1] and log[0] <= self.agent_pos[0] < log[0] + 3:
                    on_log = True
                    break

            if not on_log:
                reward = -10
                terminated = True
        
        # Dotarcie do celu (góra ekranu)
        if self.agent_pos[1] == 0:
            reward = 10
            terminated = True
        
        return self._get_obs(), reward, terminated, False, {}
    
    def _get_obs(self):
        grid = np.zeros((self.height, self.width), dtype=np.int32)

        # Pola drogi
        for y in self.car_rows:
            for x in range(self.width):
                grid[y, x] = 5

        grid[self.agent_pos[1], self.agent_pos[0]] = 1  # Agent

        for car in self.cars:
            if 0 <= car[1] < self.height and 0 <= car[0] < self.width:
                grid[car[1], car[0]] = 2

        for log in self.logs:
            for i in range(3):
                lx = log[0] + i
                if 0 <= lx < self.width:
                    grid[log[1], lx] = 3

        # Zaznacz wodę tam, gdzie nie ma kłody
        for y in self.water_rows:
            for x in range(self.width):
                if grid[y, x] == 0:
                    grid[y, x] = 4

        # Skały jako np. wartość 5 w macierzy
        for rock in self.rocks:
            grid[rock[1], rock[0]] = 5


        return grid

    
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


        self.screen.fill((170, 220 ,60))
        
        # Rysuj tło – puste pola
        for y in range(self.height):
            for x in range(self.width):
                pygame.draw.rect(self.screen, (125, 235, 80), (x * 40, y * 40, 40, 40))  # jasna zieleń

        # Rysuj wodę
        for y in self.water_rows:
            for x in range(self.width):
                if all(not (log[1] == y and log[0] <= x < log[0] + 3) for log in self.logs):
                    pygame.draw.rect(self.screen, (0, 180, 215), (x * 40, y * 40, 40, 40))

        
        # Rysuj drogę
        for y in self.car_rows:
            for x in range(self.width):
                pygame.draw.rect(self.screen, (160, 130, 100), (x * 40, y * 40, 40, 40))  # ciemnoszary asfalt
        
        # Rysuj samochody
        for car in self.cars:
            pygame.draw.rect(self.screen, (255, 55, 55), (car[0] * 40, car[1] * 40, 40, 40))

        for log in self.logs:
            for i in range(3):
                lx = log[0] + i
                if 0 <= lx < self.width:
                    pygame.draw.rect(self.screen, (145, 65, 30), (lx * 40, log[1] * 40, 40, 40))

        # Rysuj skały
        for rock in self.rocks:
            pygame.draw.rect(self.screen, (100, 100, 100), (rock[0] * 40, rock[1] * 40, 40, 40))

        
        pygame.draw.circle(self.screen, (255, 200, 0), (self.agent_pos[0] * 40 + 20, self.agent_pos[1] * 40 + 20), 20)
        
        pygame.display.flip()
        self.clock.tick(12)  # 10 FPS
    
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = CrossyRoadEnv()
    obs, _ = env.reset()
    for _ in range(250):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()
    env.close()
