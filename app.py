import numpy as np
import pandas as pd
import scipy.optimize as opt
import requests

# Get FPL data from the official API
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
data = requests.get(url).json()

# Extract player info
players = pd.DataFrame(data['elements'])

# Select relevant features
players = players[['id', 'first_name', 'second_name', 'team', 'now_cost', 'total_points', 'ep_next', 'minutes', 'element_type']]

# Rename 'ep_next' to 'expected_points' for consistency
players.rename(columns={'ep_next': 'expected_points'}, inplace=True)

# Convert cost from £ to FPL format (divide by 10)
players['now_cost'] = pd.to_numeric(players['now_cost'], errors='coerce') / 10
players['expected_points'] = pd.to_numeric(players['expected_points'], errors='coerce')

# Filter out players with missing values
players = players.dropna(subset=['now_cost', 'expected_points'])

# Get top players for each position to ensure we have enough valid candidates
def get_top_players_by_position(df, position, n=8):
    return df[df['element_type'] == position].nlargest(n, 'expected_points')

# Select players pool
gks = get_top_players_by_position(players, 1, 4)  # Goalkeepers
defs = get_top_players_by_position(players, 2, 8)  # Defenders
mids = get_top_players_by_position(players, 3, 8)  # Midfielders
fwds = get_top_players_by_position(players, 4, 6)  # Forwards

# Combine all selected players
players_selected = pd.concat([gks, defs, mids, fwds])
num_players = len(players_selected)

# Simulate performance data
np.random.seed(42)
points_matrix = np.random.randint(2, 15, size=(num_players, num_players))
points_df = pd.DataFrame(points_matrix, columns=[f'GW{i+1}' for i in range(num_players)])

# Compute variance and covariance matrix
players_selected['variance'] = points_df.var(axis=1)
cov_matrix = points_df.cov()

def squad_objective(x, expected_points, cov_matrix, alpha=0.1):
    """
    Objective function to maximize expected points while minimizing risk.
    """
    x = np.array(x, dtype=float)
    total_expected_points = np.dot(x, expected_points)
    
    # Get selected players
    selected_players = np.where(x > 0.5)[0]
    
    if len(selected_players) < 2:
        return 1e6  # High penalty for invalid solutions
        
    # Calculate risk using selected players
    selected_cov = cov_matrix.values[np.ix_(selected_players, selected_players)]
    
    try:
        # Use trace instead of determinant for more stable risk measure
        risk_penalty = np.trace(selected_cov)
    except (np.linalg.LinAlgError, ValueError):
        return 1e6
        
    return -(total_expected_points - alpha * risk_penalty)

def position_constraint(x, positions, target, pos_type):
    """
    Ensure the correct number of players in a specific position.
    """
    return np.sum(x[np.array(positions) == pos_type]) - target

def team_constraint(x, teams, team_id):
    """
    Ensure no more than 3 players from any single team.
    """
    return 3 - np.sum(x[np.array(teams) == team_id])

# Create better initial guess
def create_initial_guess():
    x0 = np.zeros(num_players)
    # Select 2 GKs
    gk_indices = np.where(players_selected['element_type'] == 1)[0]
    x0[gk_indices[:2]] = 1
    # Select 5 DEFs
    def_indices = np.where(players_selected['element_type'] == 2)[0]
    x0[def_indices[:5]] = 1
    # Select 5 MIDs
    mid_indices = np.where(players_selected['element_type'] == 3)[0]
    x0[mid_indices[:5]] = 1
    # Select 3 FWDs
    fwd_indices = np.where(players_selected['element_type'] == 4)[0]
    x0[fwd_indices[:3]] = 1
    return x0

# Define constraints
budget = 100  # Max budget
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 15},  # Total 15 players
    {'type': 'eq', 'fun': lambda x: position_constraint(x, players_selected['element_type'], 2, 1)},  # 2 GKs
    {'type': 'eq', 'fun': lambda x: position_constraint(x, players_selected['element_type'], 5, 2)},  # 5 DEFs
    {'type': 'eq', 'fun': lambda x: position_constraint(x, players_selected['element_type'], 5, 3)},  # 5 MIDs
    {'type': 'eq', 'fun': lambda x: position_constraint(x, players_selected['element_type'], 3, 4)},  # 3 FWDs
    {'type': 'ineq', 'fun': lambda x: budget - np.dot(x, players_selected['now_cost'].values)}  # Budget
]

# Add team constraints for each unique team
unique_teams = players_selected['team'].unique()
for team_id in unique_teams:
    constraints.append({
        'type': 'ineq',
        'fun': lambda x, team=team_id: team_constraint(x, players_selected['team'].values, team)
    })

# Initial guess and bounds
x0 = create_initial_guess()
bounds = [(0, 1) for _ in range(num_players)]

# Run optimization with multiple attempts
best_result = None
best_value = float('inf')

for _ in range(5):  # Try multiple times with different random perturbations
    try:
        result = opt.minimize(
            squad_objective,
            x0 + np.random.normal(0, 0.1, len(x0)),  # Add small random noise
            args=(players_selected['expected_points'].values, cov_matrix),
            constraints=constraints,
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 1000}
        )
        
        if result.fun < best_value and result.success:
            best_result = result
            best_value = result.fun
    except:
        continue

# Extract best squad
if best_result is not None:
    best_squad_indices = np.where(best_result.x > 0.5)[0]
    best_squad = players_selected.iloc[best_squad_indices]
    
    # Display results
    print("\nOptimized Squad:")
    print(best_squad[['first_name', 'second_name', 'team', 'element_type', 'now_cost', 'expected_points']])
    
    # Calculate total cost and expected points
    total_cost = best_squad['now_cost'].sum()
    total_points = best_squad['expected_points'].sum()
    print(f"\nTotal Cost: £{total_cost:.1f}m")
    print(f"Total Expected Points: {total_points:.1f}")
    
    # Display team distribution
    print("\nPlayers per team:")
    team_counts = best_squad['team'].value_counts()
    print(team_counts)
    
    # Save the optimized squad to a CSV
    best_squad.to_csv('best_squad.csv', index=False)
else:
    print("Optimization failed to find a valid solution")