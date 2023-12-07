# TensorFlow.js 是一个用于使用 JavaScript 进行机器学习开发的库

使用 JavaScript 开发机器学习模型，并直接在浏览器或 Node.js 中使用机器学习模型

我们下面介绍一些简单的操作和几个例子

首先引入我们的资源库

    //在浏览器端我们可以通过最简单的script标签进行引入
	//这是我们的tensorFlow.js库
	<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
​	//这是tensorFlow.js用来可视化训练的库

    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis"></script>

## 线性回归

我们采用最简单的数值对应的线性回归来进行入门使用

const run=async()=>{

​		const xs=[1,2,3,4]		//设置我们训练的输入值

​		const ys=[10,20,30,40]	//这组数据可以理解上分输入值的标准答案

​		 tfvis.render.scatterplot(	//这里对我们的训练数据进行可视化操作

  			{ name: '线性回归训练集' },

  			{ values: xs.map((x, i) => ({ x, y: ys[i] })) },

 			 { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }

 		);

​		const model = tf.sequential() //这里定义我们的训练模型，训练模型有序贯模型和函数模型，一般都使用我们的序贯模型

​		序贯模型会依据添加的神经网络将数据层层传递

​		model.add(tf.layers.dense({ units:1,inputShape,[1]}))//这里添加了一个全连接神经网络 输出层是一层输入层也为一层

​		model.compile({ loss:tf.losses.meanSquaredError,optimizer:tf.train.sgd(0.1)})//这里定义了我们模型的训练时采用的优化器和损失函数

​	优化器：可以理解为对模型训练的优化手段，他们采用了不同的方式对模型进行计算训练，这里用什么都行，他们最终的结果是一样的，只是过程的快慢对于某些特定的场景不一样

​	损失函数：可以对我们丢失的数据进行在处理整合，可以根据特点的场景设定不同的函数

​	//我们的tensorflow是基于我们的张量对模型进行训练，所以这里我们需要把我们的输入数据转换成张量，还有我们提供的标准答案

​		const inputs = tf.tensor(xs) 

​		const labels = tf.tensor(ys)

​	//下面使用我们模型的fit函数对我们模型进行训练

​		await model.fit(inputs,labels,{

​			batchSize:4,//这里指我们每次训练是一组数据的数量

​			epochs:200,//这里指我们的训练次数

​			callbacks:tfvis.show.fitCallbacks({	//这里将我们的训练过程可视化

​				{ name:'训练过程' },

​				['loss']

​			})

​		})

​		//这里使用模型的predict函数可以去调用我们的模型，输入我们的想要让模型帮我们寻找答案的参数

​		const output = model.predict(tf.tensor([5]))

​		//最后调用返回对象的dataSync()方法的下标为0的值就是我们想要得到的值

​		console.log('如果x为5，那么预测y为${output.dataSync()[0]}')

}

