# %%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

WATER_SIZE=3

class CrossyRoadEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 35
        self.height = 20
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.height, self.width), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        
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

        reserved_rows = 4  # Pierwsze 4 rzędy bez przeszkód
        all_rows = list(range(reserved_rows, self.height - 1))

        # Losuj rzędy z wodą (kłodami)
        self.water_rows = set(random.sample(all_rows, 3))

        # Losuj rzędy z autami, wyklucz rzędy wodne
        possible_car_rows = list(set(all_rows) - self.water_rows)
        self.car_rows = set(random.sample(possible_car_rows, 5))  # max 5 rzędów z autami

        # Twórz auta – po 1 na rząd
        for row in self.car_rows:
            x = random.randint(0, self.width - 1)
            direction = random.choice([-1, 1])
            self.cars.append([x, row, direction])

        # Twórz kłody – po 1–2 na rząd (długość 3)
        for row in self.water_rows:
            direction = random.choice([-1, 1])
            for _ in range(random.randint(1, 2)):
                start_x = random.randint(0, self.width - 3)
                self.logs.append([start_x, row, direction])

        return self._get_obs(), {}


    
    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}
        
        # Ruch agenta
        x, y = self.agent_pos
        if action == 0 and y > 0:  # Góra
            y -= 1
        # elif action == 1 and y < self.height - 1:  # Dół
            # y += 1
        elif action == 2 and x > 0:  # Lewo
            x -= 1
        elif action == 3 and x < self.width - 1:  # Prawo
            x += 1
        
        self.agent_pos = [x, y]
        
        # Ruch samochodów i kłód
        for car in self.cars:
            car[0] += car[2]
            if car[0] < 0 or car[0] >= self.width:
                car[2] *= -1
        
        for log in self.logs:
            log[0] += log[2]
            if log[0] < 0 or log[0] >= self.width:
                log[2] *= -1
        
        # Sprawdź kolizje
        reward = -0.1  # Mała kara za każdy krok
        terminated = False
        
        # Kolizja z samochodem
        for car in self.cars:
            if self.agent_pos == [car[0], car[1]]:
                reward = -10
                terminated = True
        
        # Kolizja z wodą (jeśli nie na kłodzie)
        if self.agent_pos[1] in self.water_rows:# Obszar rzeki
            on_log = False
            for log in self.logs:
                if self.agent_pos == [log[0], log[1]]:
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

        return grid

    
    def render(self):
        self.screen.fill((255, 255, 255))
        
        # Rysuj siatkę
        for y in self.water_rows:
            for x in range(self.width):
                if all(not (log[1] == y and log[0] <= x < log[0] + 3) for log in self.logs):
                    pygame.draw.rect(self.screen, (0, 100, 255), (x * 40, y * 40, 40, 40))

        
        # Rysuj agenta (kurczaka)
        pygame.draw.rect(self.screen, (255, 200, 0), (self.agent_pos[0] * 40, self.agent_pos[1] * 40, 40, 40))
        
        # Rysuj samochody
        for car in self.cars:
            pygame.draw.rect(self.screen, (255, 0, 0), (car[0] * 40, car[1] * 40, 40, 40))
        
        # Rysuj kłody
        # for log in self.logs:
        #     pygame.draw.rect(self.screen, (139, 69, 19), (log[0] * 40, log[1] * 40, 40, 40))

        for log in self.logs:
            for i in range(3):
                lx = log[0] + i
                if 0 <= lx < self.width:
                    pygame.draw.rect(self.screen, (139, 69, 19), (lx * 40, log[1] * 40, 40, 40))

        
        # Rysuj wodę
        # Rysuj wodę tylko w wierszach oznaczonych jako wodne
        for y in self.water_rows:
            for x in range(self.width):
                # Tylko jeśli nie jest pokryte kłodą – rysuj wodę
                if all(not (log[1] == y and log[0] <= x < log[0] + 3) for log in self.logs):
                    pygame.draw.rect(self.screen, (0, 100, 255), (x * 40, y * 40, 40, 40))


        
        
        pygame.display.flip()
        self.clock.tick(12)  # 10 FPS
    
    def close(self):
        pygame.quit()

# Przykład użycia
env = CrossyRoadEnv()
obs, _ = env.reset()
for _ in range(250):
    action = env.action_space.sample()  # Losowa akcja
    obs, reward, done, _, _ = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()
env.close()


