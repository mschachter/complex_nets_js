// Author: Mike Schachter
// This file contains functionality to create complex-valued recurrent networks
var cnet = {};

// define the cnet namespace
cnet._construct = function() {

	//zigg is a Gaussian random number generator
	zigg = new Ziggurat();

	// the logistic sigmoid curve
	sigmoid = function(x)
	{
		return 1.0 / (1.0 + math.pow(math.E, -x));
	}

	// class definition and constructor for a complex-valued recurrent network
	function ComplexNetwork(numNeurons, connectionProbability, weightScale, biasScale)
	{
		this.numNeurons = numNeurons;
		this.connectionProbability = connectionProbability || 0.10;
		this.weightScale = weightScale || 0.5;
		this.biasScale = biasScale || 0.1;

		// initialize the state of the network to random complex numbers
		this.state = math.zeros([numNeurons]);
		for (var k = 0; k < numNeurons; k++) {		
			// construct a random complex initial state, starting with amplitude
			var r = math.random();
			var phi = math.random()*2*math.PI;
			var z = math.complex(r*math.cos(phi), r*math.sin(phi));
			this.state[k] = z;
		}

		// initialize a real-valued Gaussian random weight matrix
		this.W = math.zeros([numNeurons, numNeurons]);
		var absmax = -1;
		for (var k = 0; k < numNeurons; k++) {
			for (var j = 0; j < numNeurons; j++) {
				if (math.random() < connectionProbability) {					
					this.W[k][j] = zigg.nextGaussian();
					if (math.abs(this.W[k][j]) > absmax) {
						absmax = math.abs(this.W[k][j]);
					}
				}
			}
		}

		// rescale the matrix
		for (var k = 0; k < numNeurons; k++) {
			for (var j = 0; j < numNeurons; j++) {
				this.W[k][j] = this.weightScale*(this.W[k][j] / absmax);
			}
		}

		// initialize real-valued bias weights
		this.bias = math.zeros([numNeurons]);
		for (var k = 0; k < numNeurons; k++) {
			this.bias[k] = zigg.nextGaussian()*this.biasScale;
		}

		// initialize an attractor, which is the location of the mouse
		this.attractor = math.complex(0, 0);
		this.attractor_weight = 0.8;
	};

	//run the network for a single time step
	ComplexNetwork.prototype.step = function()
	{
		var newState = math.zeros([this.numNeurons]);
		for (var k = 0; k < this.numNeurons; k++) {			
			// compute the complex-valued activation for neuron k			
			var a = math.dot(this.W[k], this.state);
			
			//compute the bias term as weighted combination of attractor and random bias
			var u = (1.0 - this.attractor_weight)*this.bias[k];			
			u = math.add(u, math.multiply(this.attractor_weight, this.attractor));
						
			// add the overall bias term to the activation
			var ap = math.add(a, u).toPolar();

			// run the amplitude through a sigmoid
			var r = sigmoid(ap.r);

			// construct the new state
			newState[k] = math.complex(r*math.cos(ap.phi), r*math.sin(ap.phi));
		}
		this.state = newState;
	};

	var randomColor = function()
	{
		var rgb = math.randomInt([3], 255);
		return "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
	}

	this.ComplexNetwork = ComplexNetwork;
	this.randomColor = randomColor;
}

cnet._construct();
