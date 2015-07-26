
var _			= require("underscore");
var toolset		= require("toolset");
var cliTable	= require('cli-table');

var neuralnet = function(settings) {
	this.settings	= settings;				// Settings (learning rate, topology...)
	this.data		= {};					// Container
	
	/// Setup the net's structure
	this.init();
};
neuralnet.prototype.init = function() {
	
	/*
		Here we setup the net's data and we precalculate some data to speed up the maths.
	*/
	
	// Calculate the total number of neurons
	var neuronCount	= 0;
	neuronCount		+= this.settings.topology.input;	// Inputs
	neuronCount		+= this.settings.topology.output;	// Outputs
	_.each(this.settings.topology.hidden, function(hiddenLayer) {
		neuronCount	+= hiddenLayer;
	});
	
	// Pre-calculate the layers topology (unfold the input topology to speed up data access)
	var layersTopology	= [];
	layersTopology.push(this.settings.topology.input);
	_.each(this.settings.topology.hidden, function(hiddenLayer, n) {
		layersTopology.push(hiddenLayer);
	});
	layersTopology.push(this.settings.topology.output);
	
	// Precalculate the data range for each layer
	var layersRanges	= [];
	var cursor 			= 0;
	_.each(layersTopology, function(layerSize, n) {
		if (n==layersTopology.length-1) {
			return false;
		}
		if (n==0) {
			layersRanges.push([0, layerSize]);
			cursor += layerSize;
			layersRanges.push([cursor, cursor+layersTopology[n+1]]);
		} else {
			cursor += layerSize;
			layersRanges.push([cursor, cursor+layersTopology[n+1]]);
		}
	});
	
	// Create the net's data structure
	this.net		= {
		size:			neuronCount,
		layers:			{
			topology:	layersTopology,
			ranges:		layersRanges,
			count:		layersTopology.length
		},
		weights:		[],	// Weight is a 2D array [neuronCount][neuronCount]
		values:			new Float32Array(neuronCount),
		errors:			new Float32Array(neuronCount)
		//thresholds:		new Float32Array(neuronCount)
	};
	
	// Init the values
	var i,j;
	for (i=0;i<this.net.size;i++) {
		this.net.weights.push(new Float32Array(this.net.size));
		for (j=0;j<this.net.size;j++) {
			this.net.weights[i][j]	= Math.random()*2;	// Will require optimization to use less RAM when working on large networks
		}
		this.net.values[i]		= Math.random()/Math.random();
		//this.net.thresholds[i]	= Math.random()/Math.random();
	}
	
	toolset.log("this.net",this.net);
	
	//toolset.log("this.getWeight(0,0,1,0)",this.getWeight(0,0,1,0));
}

/*
	layer
	map() for layers for fast r/w ops
*/
neuralnet.prototype.layer = function(name, n, callback) {
	var i, c;
	c = 0;
	for (i=this.net.layers.ranges[n][0];i<this.net.layers.ranges[n][1];i++) {
		this.net[name][i] = callback(this.net[name][i], c, i);	// value, zeroIndex, realIndex
		c++;
	}
}
neuralnet.prototype.netXY = function(name, x, y) {
	return this.net[name][this.net.layers.ranges[x][0]+y];
}
neuralnet.prototype.netSet = function(name, x, y, value) {
	this.net[name][this.net.layers.ranges[x][0]+y] = value;
}
neuralnet.prototype.getWeight = function(x1, y1, x2, y2) {
	return this.net.weights[this.net.layers.ranges[x1][0]+y1][this.net.layers.ranges[x2][0]+y2];
}
neuralnet.prototype.setWeight = function(x1, y1, x2, y2, value) {
	this.net.weights[this.net.layers.ranges[x1][0]+y1][this.net.layers.ranges[x2][0]+y2] = value;
}
neuralnet.prototype.incWeight = function(x1, y1, x2, y2, value) {
	this.net.weights[this.net.layers.ranges[x1][0]+y1][this.net.layers.ranges[x2][0]+y2] += value;
}
neuralnet.prototype.trainingData = function(data) {
	// Save the training data
	this.data.training	= data;
}
neuralnet.prototype.activationFunction = function(t) {
	// Sigmoid by default;
	return 1/(1+Math.pow(2.71828, 0-t));
}
neuralnet.prototype.train = function(pass) {
	var i,j;
	var error;
	for (i=0;i<pass;i++) {
		for (j=0;j<this.data.training.length;j++) {
			this.learn(this.data.training[j]);
		}
	}
}
neuralnet.prototype.evaluate = function() {
	var j;
	var output;
	
	var table	= new cliTable();
	var lines	= [];
	var cols	= [];
	
	for (i=0;i<this.data.training.length;i++) {
		cols	= [];
		cols.push('input');
		// Push the inputs
		for (j=0;j<this.data.training[i][0].length;j++) {
			cols.push(this.data.training[i][0][j]);
		}
		// Push a separator
		cols.push('expect');
		// Push the expected output
		for (j=0;j<this.data.training[i][1].length;j++) {
			cols.push(this.data.training[i][1][j]);
		}
		// Push a separator
		cols.push('got');
		output = this.test(this.data.training[i][0], true);
		for (j=0;j<output.length;j++) {
			cols.push(output[j]);
		}
		table.push(cols);
		
	}
	
	console.log(table.toString());
}

neuralnet.prototype.test = function(input, interpret) {
	var i;
	var l = input.length;
	// Save the input into the runtime data
	for (i=0;i<l;i++) {
		this.netSet('values', 0, i, input[i]);
	}
	
	var p, p1;
	// Only iterate on the hidden layers and the output, skip the input
	// For each layer after the input
	for (i=1;i<this.net.layers.count;i++) {
		// For each neuron in the current layer
		for (p=0;p<this.net.layers.topology[i];p++) {
			var out = 0;
			// For each neuron in the previous layer
			for (p1=0;p1<this.net.layers.topology[i-1];p1++) {
				out += this.netXY('values', i-1, p1)*this.getWeight(i-1, p1, i, p);
			}
			// Save the value
			this.netSet('values', i, p, this.activationFunction(out));
		}
	}
	//toolset.log("this.data.values",this.data.values);
	
	// return the output
	var output = [];
	for (i=0;i<this.net.layers.topology[this.net.layers.count-1];i++) {
		output.push(this.netXY('values', this.net.layers.count-1, i));
	}
	
	//console.log("output -> ", output);
	
	
	/*if (interpret) {
		if (output<=0.4) {
			return 0;
		}
		if (output>=0.6) {
			return 1;
		}
		return 'ir';
	}*/
	return output;	// Array of output values
}


neuralnet.prototype.learn = function(dataset) {
	var scope = this;
	//console.info("learn -> ",dataset);
	// test
	var output					= this.test(dataset[0]);
	var o, neuronIndex0, neuronIndex, neuronIndex1; // Neuron Index: previous layer, current layer, next layer
	var layer;					// Layer Index
	var absError;				// Absolute Error
	var sumSquareError = 0;		// Sum of the squared error
	var errorGradient;			// The gradient used to update the weights
	var neuronValue;			// Buffer to contain the neuron value
	var errorPropagationFactor;	// Error Propagation Factor for the hidden layers
	
	// Process the first set of weights (leading to the output layer)
	for (o=0;o<output.length;o++) {
		// We calculate the absolute error and the gradient. We also sum the root squre of the absolute error to measure the learning state.
		absError		= dataset[1][o]-output[o];
		sumSquareError	+= Math.pow(absError, 2);
		errorGradient	= output[o]*(1-output[o])*absError;
		
		// Save the error
		this.netSet('errors', this.net.layers.count-1, o, errorGradient);
		
		// Now we calculate the new weights between the output neurons and the nerons of the previous layer
		for (neuronIndex=0;neuronIndex<this.net.layers.topology[this.net.layers.count-2];neuronIndex++) {
			// Update the weight [last hidden layer;neuronIndex][outputLayer;o]
			this.incWeight(
				this.net.layers.count-2, neuronIndex,
				this.net.layers.count-1, o,
				this.settings.learning.rate*errorGradient*this.netXY('values', this.net.layers.count-2, neuronIndex)
			);
		}
	}
	
	
	// Backpropagate the error
	for (layer=this.net.layers.count-2;layer>=1;layer--) {
		for (neuronIndex=0;neuronIndex<this.net.layers.topology[layer];neuronIndex++) {
			// Calculate the error
			neuronValue		= this.netXY('values', layer, neuronIndex);
			
			// Calculate the error propagation
			errorPropagationFactor	= 0;
			for (neuronIndex1=0;neuronIndex1<this.net.layers.topology[layer+1];neuronIndex1++) {
											// Error * weight
				errorPropagationFactor	+=	this.netXY('errors', layer+1, neuronIndex1) * this.getWeight(layer, neuronIndex, layer+1, neuronIndex1);
			}
			
			// We calculate the error gradient for each of the current layer's neurons
			errorGradient	= neuronValue*(1-neuronValue)*errorPropagationFactor
			
			// We calculate the new weight
			for (neuronIndex0=0;neuronIndex0<this.net.layers.topology[layer-1];neuronIndex0++) {
				this.incWeight(
					layer-1,	neuronIndex0,
					layer,		neuronIndex,
					this.settings.learning.rate*errorGradient*this.netXY('values', layer-1, neuronIndex0)
				);
			}
		}
	}
	return sumSquareError;
}

module.exports	= neuralnet;
