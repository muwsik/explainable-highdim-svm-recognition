#%%
import numpy as np
import plotly.graph_objects as go

import linGenerator as lg

# 2D visualization
def plot2D(tempLinearSample):
    a = np.array(tempLinearSample.params["a"], dtype=float)
    b = tempLinearSample.params["b"]

    pos = tempLinearSample.X[tempLinearSample.Y == 1]
    neg = tempLinearSample.X[tempLinearSample.Y == -1]

    xmin, xmax = np.min(tempLinearSample.X[:, 0]), np.max(tempLinearSample.X[:, 0])
    ymin, ymax = np.min(tempLinearSample.X[:, 1]), np.max(tempLinearSample.X[:, 1])

    fig = go.Figure()

    # 
    fig.add_trace(go.Scatter(
        x = pos[:, 0], y = pos[:, 1],
        mode = 'markers',
        name = '+1',
        marker = dict(color = 'blue'),
        opacity = 0.6
    ))

    fig.add_trace(go.Scatter(
        x = neg[:, 0], y = neg[:, 1],
        mode = 'markers',
        name = '-1',
        marker = dict(color='red'),
        opacity = 0.6
    ))

    if abs(a[1]) >= 1e-6:
        xLine = np.linspace(xmin, xmax, 2)
        yLine = -(a[0] * xLine - b) / a[1]

        fig.add_trace(go.Scatter(
            x = xLine, y = yLine,
            mode = 'lines',
            name = 'rule',
            line = dict(color = 'black')
        ))
    else:
        fig.add_vline(x = b / a[0], line=dict(color = 'black'), name = 'rule')

    fig.update_layout(
        title = f"a = {a}, b = {b}",
        xaxis_title = "feature_1",
        yaxis_title = "feature_2"
    )

    fig.update_xaxes(range = [xmin - 1, xmax + 1])
    fig.update_yaxes(range = [ymin - 1, ymax + 1])

    return fig


# 3D visualization
def drawPlane(a, c, xLim, yLim, zLim):
    if abs(a[0]) >= 1e-6:
        Y, Z = np.meshgrid(
            np.linspace(yLim[0], yLim[1], 5),
            np.linspace(zLim[0], zLim[1], 5)
        )
        X = (c - a[1]*Y - a[2]*Z) / a[0]

    elif abs(a[1]) >= 1e-6:
        X, Z = np.meshgrid(
            np.linspace(xLim[0], xLim[1], 5),
            np.linspace(zLim[0], zLim[1], 5)
        )
        Y = (c - a[0]*X - a[2]*Z) / a[1]

    else:
        X, Y = np.meshgrid(
            np.linspace(xLim[0], xLim[1], 5),
            np.linspace(yLim[0], yLim[1], 5)
        )
        Z = (c - a[0]*X - a[1]*Y) / a[2]

    return X, Y, Z

def plot3D(tempLinearSample):
    a = np.array(tempLinearSample.params["a"], dtype = float)
    b = tempLinearSample.params["b"]

    pos = tempLinearSample.X[tempLinearSample.Y == 1]
    neg = tempLinearSample.X[tempLinearSample.Y == -1]

    xLim = [np.min(tempLinearSample.X[:, 0]), np.max(tempLinearSample.X[:, 0])]
    yLim = [np.min(tempLinearSample.X[:, 1]), np.max(tempLinearSample.X[:, 1])]
    zLim = [np.min(tempLinearSample.X[:, 2]), np.max(tempLinearSample.X[:, 2])]

    fig = go.Figure()

    # точки
    fig.add_trace(go.Scatter3d(
        x = pos[:, 0], y = pos[:, 1], z = pos[:, 2],
        mode = 'markers',
        name = '+1',
        marker = dict(color = 'blue', size = 3),
        opacity = 0.6
    ))

    fig.add_trace(go.Scatter3d(
        x = neg[:, 0], y = neg[:, 1], z = neg[:, 2],
        mode = 'markers',
        name = '-1',
        marker = dict(color = 'red', size = 3),
        opacity = 0.6
    ))

    # плоскость
    X, Y, Z = drawPlane(a, b, xLim, yLim, zLim)

    fig.add_trace(go.Surface(
        x = X,
        y = Y,
        z = Z,
        opacity = 0.3,
        showscale = False,
        hoverinfo = 'skip'
    ))

    fig.update_layout(
        scene = dict(
            xaxis = dict(
                title = 'feature_1',
                range = [xLim[0] - 1, xLim[1] + 1],
                showspikes = False
            ),
            yaxis = dict(
                title = 'feature_2',
                range = [yLim[0] - 1, yLim[1] + 1],
                showspikes = False
            ),
            zaxis = dict(
                title = 'feature_3',
                range = [zLim[0] - 1, zLim[1] + 1],
                showspikes = False
            ),
        )
    )

    return fig

  

#%% 2D generation and visualization
generateParams2D = {
    "objNum": 100,
    "halfSize": 10,
    "featNum": 2,
    "sigma": 0.8,
    "a": [1, 5],
    "b": 2
}

linGenerator = lg.LinearGenerator()
baseLinearSample = linGenerator.base(
    generateParams2D["objNum"],
    generateParams2D["featNum"],
    generateParams2D["halfSize"],
    generateParams2D["sigma"]        
)   

customLinearSample = lg.LinearGenerator().specifiedHyperplane(
    generateParams2D["objNum"],
    generateParams2D["featNum"],
    generateParams2D["halfSize"],
    generateParams2D["sigma"],
    generateParams2D["a"],
    generateParams2D["b"]        
)

plot2D(baseLinearSample).show()
plot2D(customLinearSample).show()

#%% 3D generation and visualization
generateParams3D = {
    "objNum": 100,
    "halfSize": 10,
    "featNum": 3,
    "sigma": 1,
    "a": [1, 1, 1],
    "b": -2
}

linGenerator = lg.LinearGenerator()
baseLinearSample = linGenerator.base(
    generateParams3D["objNum"],
    generateParams3D["featNum"],
    generateParams3D["halfSize"],
    generateParams3D["sigma"]        
)   

customLinearSample = lg.LinearGenerator().specifiedHyperplane(
    generateParams3D["objNum"],
    generateParams3D["featNum"],
    generateParams3D["halfSize"],
    generateParams3D["sigma"],
    generateParams3D["a"],
    generateParams3D["b"]        
)

plot3D(baseLinearSample).show()
plot3D(customLinearSample).show()


#%% TXT save and load without generate parametrs
generateParams = {
    "objNum": 100,
    "halfSize": 100,
    "featNum": 100,
    "sigma": 1,
    "a": None,  # random
    "b": -2
} 

customLinearSample = lg.LinearGenerator().specifiedHyperplane(
    generateParams["objNum"],
    generateParams["featNum"],
    generateParams["halfSize"],
    generateParams["sigma"],
    generateParams["a"],
    generateParams["b"]        
) 

customLinearSample.saveTXT(r'D:\customLinearSample.txt')

sampleFromFile = lg.Sample()
sampleFromFile.loadTXT(r'D:\customLinearSample.txt')


#%% binary save and load with generate parametrs
generateParams = {
    "objNum": 10000,
    "halfSize": 50,
    "featNum": 5000,
    "sigma": 0.8,
    "a": None,  # random
    "b": +10
} 

customLinearSample = lg.LinearGenerator().specifiedHyperplane(
    generateParams["objNum"],
    generateParams["featNum"],
    generateParams["halfSize"],
    generateParams["sigma"],
    generateParams["a"],
    generateParams["b"]        
)

customLinearSample.saveBin(r'D:\ds-10k-5k-08-rnd-10.npz')
    
sampleFromFile = lg.Sample()
sampleFromFile.loadBin(r'D:\ds-10k-5k-08-rnd-10.npz')    

print(sampleFromFile.params) # generate parametrs


# %% generate train and test datasets for one *a*
custon_a = np.random.uniform(low = 0, high = 1, size = 5000)

trainDataset = lg.LinearGenerator().specifiedHyperplane(
    objNum = 10000,
    featNum = 5000,
    halfSize = 100,
    sigma = 0.8,
    a = custon_a,
    b = -15        
)
#trainDataset.saveBin(r"D:\datasets\ds-train-10k-5k-08-rnd--15.npz")

testDataset = lg.LinearGenerator().specifiedHyperplane(
    objNum = 5000,
    featNum = 5000,
    halfSize = 100,
    sigma = 0.8,
    a = custon_a,
    b = -15        
)
#testDataset.saveBin(r"D:\datasets\ds-test-5k-5k-08-rnd--15.npz")
    
trainDataset = lg.Sample.fromBin(r"D:\datasets\ds-train-10k-5k-08-rnd--15.npz")
print(trainDataset.params)

testDataset = lg.Sample.fromBin(r"D:\datasets\ds-test-5k-5k-08-rnd--15.npz")
print(testDataset.params)
