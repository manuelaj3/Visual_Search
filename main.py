"""Made by Manuel Alcántara Jurado

This code has the implementation of some Searching methods 
which are represented on a grid, selecting each method by
pressing on the keyboard the letter corresponding to each 
method.

The base of this code is copied from a Youtube video that
the teacher showed us about A* searching method, and based 
on it , I added the other methods.

"""

import pygame  
import math  
from queue import PriorityQueue, Queue, LifoQueue  

WIDTH = 800  # Set width of the window
WIN = pygame.display.set_mode((WIDTH, WIDTH))  
pygame.display.set_caption("Path Finding Algorithms")  

RED = (255, 0, 0)  # Represents closed nodes
GREEN = (0, 255, 0)  # Represents open nodes
BLUE = (0, 255, 0)  
YELLOW = (255, 255, 0)  
WHITE = (255, 255, 255)  # Represents unvisited nodes
BLACK = (0, 0, 0)  # Represents barriers
PURPLE = (128, 0, 128)  # Represents the path
ORANGE = (255, 165, 0)  # Represents start node
GREY = (128, 128, 128)  # Represents grid lines
TURQUOISE = (64, 224, 208)  # Represents end node

# Spot class to represent each cell in the grid
class Spot:
    def __init__(self, row, col, width, total_rows):  # Initialize spot with position and size
        self.row = row  # Row position in grid
        self.col = col  # Column position in grid
        self.x = row * width  # Pixel x-coordinate
        self.y = col * width  # Pixel y-coordinate
        self.color = WHITE  # Default color is white (unvisited)
        self.neighbors = []  # List to store neighboring nodes
        self.width = width  # Width of each cell
        self.total_rows = total_rows  # Total number of rows in the grid

    def get_pos(self):  # Get the grid position of the spot
        return self.row, self.col

    def is_closed(self):  # Check if the node is closed (red)
        return self.color == RED

    def is_open(self):  # Check if the node is open (green)
        return self.color == GREEN

    def is_barrier(self):  # Check if the node is a barrier (black)
        return self.color == BLACK

    def is_start(self):  # Check if the node is the start (orange)
        return self.color == ORANGE

    def is_end(self):  # Check if the node is the end (turquoise)
        return self.color == TURQUOISE

    def reset(self):  # Reset the node to default color (white)
        self.color = WHITE

    def make_start(self):  # Set the node as start (orange)
        self.color = ORANGE

    def make_closed(self):  # Set the node as closed (red)
        self.color = RED

    def make_open(self):  # Set the node as open (green)
        self.color = GREEN

    def make_barrier(self):  # Set the node as a barrier (black)
        self.color = BLACK

    def make_end(self):  # Set the node as end (turquoise)
        self.color = TURQUOISE

    def make_path(self):  # Set the node as part of the path (purple)
        self.color = PURPLE

    def draw(self, win):  # Draw the spot on the window with its color
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):  # Update neighbors of the node (avoid barriers)
        self.neighbors = []  # Reset neighbors list
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):  # Define less-than operator (necessary for PriorityQueue)
        return False

# Heuristic function for A* algorithm
def h(p1, p2):
    x1, y1 = p1  # First point coordinates
    x2, y2 = p2  # Second point coordinates
    return abs(x1 - x2) + abs(y1 - y2)  # Return Manhattan distance

# Print the path found by the algorithm
def print_path(came_from, current):
    path = []  # Initialize path list
    while current in came_from:  # Traverse through the came_from dictionary
        path.append(current.get_pos())  # Append position to path
        current = came_from[current]  # Move to previous node
    path = path[::-1]  # Reverse path order
    print("Path")  # Print path header
    print("Start:")
    for coord in path:
        print(f"  {coord},")  # Print each coordinate
    print("Goal")
    print(f"Length: {len(path)} steps")  # Print path length

# Reconstruct the path visually on the grid
def reconstruct_path(came_from, current, draw):
    while current in came_from:  # Traverse came_from dictionary
        current = came_from[current]  # Move to previous node
        current.make_path()  # Mark as part of the path
        draw()  # Draw the grid

# A* pathfinding algorithm implementation
def a_star(draw, grid, start, end):
    count = 0  # Counter to keep track of insertion order
    open_set = PriorityQueue()  # Priority queue for A* algorithm
    open_set.put((0, count, start))  # Initialize with start node
    came_from = {}  # Dictionary to store paths
    g_score = {spot: float("inf") for row in grid for spot in row}  # Initialize g-scores to infinity
    g_score[start] = 0  # Start node g-score is 0
    f_score = {spot: float("inf") for row in grid for spot in row}  # Initialize f-scores to infinity
    f_score[start] = h(start.get_pos(), end.get_pos())  # f-score for start node
    open_set_hash = {start}  # Hash set for open set

    while not open_set.empty():  # Main loop for A*
        for event in pygame.event.get():  # Event handling to quit
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]  # Get the node with lowest f-score
        open_set_hash.remove(current)  # Remove current node from open set hash

        if current == end:  # Check if end node reached
            reconstruct_path(came_from, end, draw)  # Reconstruct path visually
            print_path(came_from, end)  # Print path
            end.make_end()  # Mark end node
            return True

        for neighbor in current.neighbors:  # Process neighbors
            temp_g_score = g_score[current] + 1  # Calculate temporary g-score

            if temp_g_score < g_score[neighbor]:  # Check if path is better
                came_from[neighbor] = current  # Update path to neighbor
                g_score[neighbor] = temp_g_score  # Update g-score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())  # Update f-score
                if neighbor not in open_set_hash:  # Add neighbor if not in open set
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))  # Add with priority
                    open_set_hash.add(neighbor)  # Add to hash set
                    neighbor.make_open()  # Mark as open

        draw()  # Draw updated grid
        if current != start:
            current.make_closed()  # Mark current node as closed

    return False

# Breadth-First Search (BFS) pathfinding algorithm
def bfs(draw, grid, start, end):
    queue = Queue()  # Initialize FIFO queue for BFS
    queue.put(start)  # Add the start node to the queue
    came_from = {}  # Dictionary to track the path
    visited = {start}  # Set to track visited nodes

    while not queue.empty():  # Main BFS loop
        for event in pygame.event.get():  # Handle pygame quit events
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.get()  # Dequeue the next node

        if current == end:  # Check if we reached the end node
            reconstruct_path(came_from, end, draw)  # Visually reconstruct the path
            print_path(came_from, end)  # Print the path coordinates
            end.make_end()  # Mark the end node
            return True

        for neighbor in current.neighbors:  # Iterate over neighboring nodes
            if neighbor not in visited:  # If neighbor hasn't been visited
                came_from[neighbor] = current  # Update the path
                visited.add(neighbor)  # Mark as visited
                queue.put(neighbor)  # Add to the queue
                neighbor.make_open()  # Mark as open

        draw()  # Draw updated grid
        if current != start:
            current.make_closed()  # Mark current node as closed

    return False  # Return False if no path is found

# Depth-First Search (DFS) pathfinding algorithm
def dfs(draw, grid, start, end):
    stack = LifoQueue()  # Initialize LIFO stack for DFS
    stack.put(start)  # Add the start node to the stack
    came_from = {}  # Dictionary to track the path
    visited = {start}  # Set to track visited nodes

    while not stack.empty():  # Main DFS loop
        for event in pygame.event.get():  # Handle pygame quit events
            if event.type == pygame.QUIT:
                pygame.quit()

        current = stack.get()  # Pop the next node from the stack

        if current == end:  # Check if we reached the end node
            reconstruct_path(came_from, end, draw)  # Visually reconstruct the path
            print_path(came_from, end)  # Print the path coordinates
            end.make_end()  # Mark the end node
            return True

        for neighbor in current.neighbors:  # Iterate over neighboring nodes
            if neighbor not in visited:  # If neighbor hasn't been visited
                came_from[neighbor] = current  # Update the path
                visited.add(neighbor)  # Mark as visited
                stack.put(neighbor)  # Add to the stack
                neighbor.make_open()  # Mark as open

        draw()  # Draw updated grid
        if current != start:
            current.make_closed()  # Mark current node as closed

    return False  # Return False if no path is found

# Uniform Cost Search (UCS) pathfinding algorithm
def ucs(draw, grid, start, end):
    open_set = PriorityQueue()  # Priority queue for UCS
    open_set.put((0, start))  # Start node with cost 0
    came_from = {}  # Dictionary to track the path
    cost = {spot: float("inf") for row in grid for spot in row}  # Initialize costs to infinity
    cost[start] = 0  # Cost of start node is 0

    while not open_set.empty():  # Main UCS loop
        for event in pygame.event.get():  # Handle pygame quit events
            if event.type == pygame.QUIT:
                pygame.quit()

        current_cost, current = open_set.get()  # Get node with lowest cost

        if current == end:  # Check if we reached the end node
            reconstruct_path(came_from, end, draw)  # Visually reconstruct the path
            print_path(came_from, end)  # Print the path coordinates
            end.make_end()  # Mark the end node
            return True

        for neighbor in current.neighbors:  # Iterate over neighboring nodes
            new_cost = current_cost + 1  # Calculate cost to neighbor
            if new_cost < cost[neighbor]:  # Check if path is cheaper
                came_from[neighbor] = current  # Update the path
                cost[neighbor] = new_cost  # Update cost
                open_set.put((new_cost, neighbor))  # Add to priority queue
                neighbor.make_open()  # Mark as open

        draw()  # Draw updated grid
        if current != start:
            current.make_closed()  # Mark current node as closed

    return False  # Return False if no path is found

# Dijkstra's algorithm (wrapper for UCS, as they are identical here)
def dijkstra(draw, grid, start, end):
    return ucs(draw, grid, start, end)  # Call UCS as Dijkstra is the same

# Function to create a grid of Spot objects
def make_grid(rows, width):
    grid = []  # Initialize grid as a list
    gap = width // rows  # Calculate size of each cell
    for i in range(rows):
        grid.append([])  # Add new row to grid
        for j in range(rows):
            spot = Spot(i, j, gap, rows)  # Create a new Spot object
            grid[i].append(spot)  # Add Spot to grid row
    return grid  # Return the completed grid

# Draw grid lines on the window
def draw_grid(win, rows, width):
    gap = width // rows  # Calculate size of each cell
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))  # Draw horizontal lines
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))  # Draw vertical lines

# Draw entire window including grid and cells
def draw(win, grid, rows, width):
    win.fill(WHITE)  # Fill background with white
    for row in grid:
        for spot in row:
            spot.draw(win)  # Draw each Spot on the grid
    draw_grid(win, rows, width)  # Draw grid lines
    pygame.display.update()  # Update the display

# Convert mouse click position to grid coordinates
def get_clicked_pos(pos, rows, width):
    gap = width // rows  # Calculate size of each cell
    y, x = pos  # Get mouse click coordinates
    row = y // gap  # Calculate row in grid
    col = x // gap  # Calculate column in grid
    return row, col  # Return grid position as row, column

# Main function to run the program
def main(win, width):
    ROWS = 50  # Define the number of rows in the grid
    grid = make_grid(ROWS, width)  # Create the grid

    start = None  # Variable for start node
    end = None  # Variable for end node
    run = True  # Control loop for the program
    while run:
        draw(win, grid, ROWS, width)  # Draw the grid
        for event in pygame.event.get():  # Handle events
            if event.type == pygame.QUIT:  # Check for quit event
                run = False  # Exit the loop

            if pygame.mouse.get_pressed()[0]:  # LEFT mouse button
                pos = pygame.mouse.get_pos()  # Get mouse position
                row, col = get_clicked_pos(pos, ROWS, width)  # Get grid coordinates
                spot = grid[row][col]  # Get Spot at coordinates
                if not start and spot != end:  # If start not defined and not the end node
                    start = spot
                    start.make_start()  # Mark Spot as start
                elif not end and spot != start:  # If end not defined and not the start node
                    end = spot
                    end.make_end()  # Mark Spot as end
                elif spot != end and spot != start:  # If not start or end, mark as barrier
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT mouse button
                pos = pygame.mouse.get_pos()  # Get mouse position
                row, col = get_clicked_pos(pos, ROWS, width)  # Get grid coordinates
                spot = grid[row][col]  # Get Spot at coordinates
                spot.reset()  # Reset spot to default
                if spot == start:  # If resetting start
                    start = None
                elif spot == end:  # If resetting end
                    end = None

            if event.type == pygame.KEYDOWN:  # Keyboard input handling
                if event.key == pygame.K_a and start and end:  # If 'A' pressed
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)  # Update neighbors for A* search
                    a_star(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_d and start and end:  # If 'D' pressed
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)  # Update neighbors for DFS
                    dfs(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_b and start and end:  # If 'B' pressed
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)  # Update neighbors for BFS
                    bfs(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_u and start and end:  # If 'U' pressed
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)  # Update neighbors for UCS
                    ucs(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_r and start and end:  # If 'R' pressed
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)  # Update neighbors for Dijkstra
                    dijkstra(lambda: draw(win, grid, ROWS, width), grid, start, end)

                if event.key == pygame.K_c:  # If 'C' pressed, clear grid
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)  # Reset grid

    pygame.quit()  # Quit pygame

main(WIN, WIDTH)  # Call main function to start the program


