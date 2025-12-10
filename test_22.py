n = [2,3,4,6,8,1,5]

for i in range(len(n)):
    if i % 2 == 0:
        print(f"{i} is even")
        
        
def siddi(n):
    for i in range(len(n)):
        if i % 2 == 0:
            print(f"{i} is even")
              
siddi(n)


def agent_rag(query, context):
    print("Query:", query)
    print("Context:", context)
    
agent_rag("What is AI?", ["AI is the simulation of human intelligence by machines.", "It includes learning, reasoning, and self-correction."])