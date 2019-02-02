import random
from random import choice
import numpy as np
import time

going_right = True
going_left = False

class simple_reflex_agent():
    def __init__(self, environment):
        self.nb_moves = 0
        self.vac_location = random.randint(0, 8)
        self.nb_steps = 0
        self.sucked = 0
        self.env = environment.locationCondition
        self.nb_piles = environment.nb_dirt_piles
        self.clean = False

        print(self.env)
        print("starting location: {}".format(self.vac_location))

    def execute(self):
        # While it's not at the move right
        while self.vac_location <= 8 and self.sucked != self.nb_piles:
            if self.env[self.vac_location] == 1:
                print("suck")
                self.sucked += 1
                self.env[self.vac_location] = 0
                print("sucked total: {}".format(self.sucked))
                print(self.env)
            elif self.env[self.vac_location] == 0:
                if self.vac_location < 8:
                    print("Moving right to {}".format(self.vac_location + 1))
                    self.vac_location += 1
                    self.nb_steps += 1
                elif self.vac_location == 8:
                    while self.vac_location >= 0 and self.sucked != self.nb_piles:
                        if self.env[self.vac_location] == 1:
                            print("suck")
                            self.sucked += 1
                            self.env[self.vac_location] = 0
                            print(self.env)
                            print("sucked total: {}".format(self.sucked))
                        else:
                            print("Moving left to {}".format(self.vac_location-1))
                            self.vac_location -= 1
                            self.nb_steps += 1

        print("perf: {}".format(self.nb_steps))
        return self.nb_steps

    def murphy_vacuum(self, clean):
        if random.randrange(0, 100) < 25:
            print("sucking failed")
        else:
            # It's clean
            self.sucked +=1
            self.env[self.vac_location] = 0
            print("Sucked succesffuly")
            print("Env after suck: {}".format(self.env))

    def move_right(self):
        print("Moving right to {}.".format(self.vac_location+1))
        self.vac_location += 1
        self.nb_steps += 1

        if random.randrange(0,100)<25:
            print("Oops dropped some dirt")
            self.env[self.vac_location] = 1
            print("env after dropping shit: {}".format(self.env))

    def move_left(self):
        print("Moving left To {}.".format(self.vac_location-1))

        self.vac_location -= 1
        self.nb_steps += 1
        if random.randrange(0, 100) < 25:
            print("Oops dropped some dirt")
            self.env[self.vac_location] = 1
            print("env after dropping shit: {}".format(self.env))


    def perceive(self):
        clean = False
        print("Perceiving..")
        if self.env[self.vac_location] == 0:
            print("IT's clean")
            clean = True
            if random.randrange(0,100)< 10:
                print("But you got murphy'd so I think it's dirty")
                clean = False
        elif self.env[self.vac_location] == 1:
            print("IT's dirty")
            clean = False
            if random.randrange(0,100)< 10:
                print("But you got murphy'd so I think it's clean")
                clean = True

        return clean

    def execute_murphy(self):
        while(np.any(self.env[0:] == 1)):
            # While it's not at the move right
            while self.vac_location < 8 and np.any(self.env[0:] == 1):
                    c = self.perceive()
                    while(c != True):
                        # If it thinks dirty
                        self.murphy_vacuum(c)
                        # Check if clean
                        check = self.perceive()
                        if check: break
                    #if clean move
                    if self.vac_location == 8:
                        break
                    self.move_right()

            if self.vac_location == 8 and np.any(self.env[::-1] == 1):
                while self.vac_location >= 0:
                    # Perceive current location
                    c = self.perceive()
                    while(c != True):
                        # If it thinks dirty
                        self.murphy_vacuum(c)
                        # Check if clean
                        check = self.perceive()
                        if check: break
                    #if clean move
                    if self.vac_location == 0:
                        break
                    self.move_left()

        print("steps: {}".format(self.nb_steps))
        print("Sucked: {}".format(self.sucked))
        return self.nb_steps, self.sucked


class random_agent():
    def __init__(self, environment):
        self.nb_moves = 0
        self.vac_location = random.randint(0, 8)
        #self.nb_steps = 0
        self.sucked = 0
        self.env = environment.locationCondition
        self.nb_piles = environment.nb_dirt_piles
        print(self.env)
        print("starting location: {}".format(self.vac_location))

    def random_movement(self, curr_loc):
        if curr_loc == 0:
            next = random.choice([1,3])
        elif curr_loc==1:
            next = random.choice([0,2,4])
        elif curr_loc == 2:
            next = random.choice([1,5])
        elif curr_loc==3:
            next = random.choice([0,4,6])
        elif curr_loc==4:
            next = random.choice([1,3,5,7])
        elif curr_loc==5:
            next = random.choice([2,4,8])
        elif curr_loc==6:
            next = random.choice([3,7])
        elif curr_loc==7:
            next = random.choice([6,4,8])
        elif curr_loc==8:
            next = random.choice([7,5])

        return next

    def perceive_and_action(self, location):
        while self.sucked != self.nb_piles:
            print("Currently in location: {}".format(location))
            if self.env[location] == 1:
                print("Theres dirt! What should I do now?")
                random_action = random.choice([0,1])
                print("random_action:{}".format(random_action))
                if random_action == 0:
                    next_spot = self.random_movement(location)
                    print("nah don't feel like cleaning. moving to next random spot -> {}".format(next_spot))
                    #time.sleep(1)
                    self.nb_moves += 1
                    self.perceive_and_action(next_spot)
                elif random_action == 1:
                    print("Ok I'll clean")
                    self.env[location] = 0
                    self.sucked += 1
                    print(self.env)
                    #time.sleep(1)
            elif self.env[location] == 0:
                next_spot = self.random_movement(location)
                print("There's nothing to clean. Moving to next random spot -> {}".format(next_spot))
                #time.sleep(1)
                self.nb_moves += 1
                self.perceive_and_action(next_spot)

    def perceive(self):
        clean = False
        print("Perceiving..")
        if self.env[self.vac_location] == 0:
            print("IT's clean")
            clean = True
            if random.randrange(0, 100) < 10:
                print("But you got murphy'd so I think it's dirty")
                clean = False
        elif self.env[self.vac_location] == 1:
            print("IT's dirty")
            clean = False
            if random.randrange(0, 100) < 10:
                print("But you got murphy'd so I think it's clean")
                clean = True

        return clean

    def murphy_vacuum(self):
        if random.randrange(0, 100) < 25:
            print("sucking failed")
        else:
            # It's clean
            self.sucked += 1
            self.env[self.vac_location] = 0
            print("Sucked succesffuly")
            print("Env after suck: {}".format(self.env))

    def move_random(self, location):
        self.vac_location = location
        print("moving to: {}".format(location))
        if random.randrange(0, 100) < 25:
            print("Oops dropped some dirt")
            self.env[self.vac_location] = 1
            self.nb_moves += 1
            print("env after dropping shit: {}".format(self.env))

    def perceive_and_action_murphy(self):
        while(np.any(self.env[0:] == 1)):
            print("Current location: {}".format(self.vac_location))
            c = self.perceive()
            while(c != True):
                random_action = random.choice([0,1])
                if random_action == 0:
                    next_spot = self.random_movement(self.vac_location)
                    self.vac_location = next_spot
                    print("nah don't feel like cleaning. moving to next random spot")
                    self.move_random(next_spot)
                    c = self.perceive()
                elif random_action == 1:
                    print("OK I'll clean")
                    while(c!= True):
                        self.murphy_vacuum()
                        check = self.perceive()
                        if check: break

        return self.nb_moves, self.sucked


class environment(object):
    def __init__(self, nb_dirt_piles):
        # 0 is clean 1 is dirty
        #self.locationCondition = {'1':0,'2':0,'3':0,'4':0,'5':0, '6':0, '7':0, '8':0, '9':0}
        self.nb_dirt_piles = nb_dirt_piles
        a = np.array([0] * (9-nb_dirt_piles) + [1] * nb_dirt_piles)
        np.random.shuffle(a)
        self.locationCondition = a


def main():
    print("------------------------------------------------------------------------------------------------------------")
    total_steps = 0
    total_sucked = 0
    i = 0

    while i in range(100):
        e = environment(1)
        agent = random_agent(e)
        #p = agent.execute()
        steps, sucked = agent.perceive_and_action_murphy()

        #agent = simple_reflex_agent(e)
        #steps = agent.execute()
        #steps, sucked = agent.execute_murphy()


        total_steps += steps
        total_sucked += sucked
        i += 1

    steps_perf = total_steps/100
    sucked_perf = total_sucked/100

    print("Number of steps: {}".format(steps_perf))
    print("Number of sucked aciton: {}".format(sucked_perf))

if __name__ == '__main__':
    main()
