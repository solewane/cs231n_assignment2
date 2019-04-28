#data = get_CIFAR10_data()
#for k, v in data.items():
#  print ('%s: ' % k, v.shape)

#from cs231n.layers import affine_forward



  #############################################################################
  # test Affine layer: foward                                                 #
  #############################################################################
#
#num_inputs = 2
#input_shape = (4, 5, 6)
#output_dim = 3
#
#input_size = num_inputs * np.prod(input_shape)
#weight_size = output_dim * np.prod(input_shape)
#
#x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
#w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
#b = np.linspace(-0.3, 0.1, num=output_dim)
#
#out, _ = affine_forward(x, w, b)
#correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
#                        [ 3.25553199,  3.5141327,   3.77273342]])
#
## Compare your output with ours. The error should be around 1e-9.
#print ('Testing affine_forward function:')
#print ('difference: ', rel_error(out, correct_out))

  #############################################################################
  #                             END OF TEST                                   #
  #############################################################################



  #############################################################################
  # test Affine layer: foward                                                 #
  #############################################################################
# Test the affine_backward function
#
#x = np.random.randn(10, 2, 3)
#w = np.random.randn(6, 5)
#b = np.random.randn(5)
#dout = np.random.randn(10, 5)
#
#dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
#dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
#db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
#
#_, cache = affine_forward(x, w, b)
#dx, dw, db = affine_backward(dout, cache)
#
## The error should be around 1e-10
#print ('Testing affine_backward function:')
#print ('dx error: ', rel_error(dx_num, dx))
#print ('dw error: ', rel_error(dw_num, dw))
#print ('db error: ', rel_error(db_num, db))
  #############################################################################
  #                             END OF TEST                                   #
  #############################################################################
  
  
  
  # Test the relu_forward function

#x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
#
#out, _ = relu_forward(x)
#correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
#                        [ 0.,          0.,          0.04545455,  0.13636364,],
#                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])
#
## Compare your output with ours. The error should be around 1e-8
#print ('Testing relu_forward function:')
#print ('difference: ', rel_error(out, correct_out))
  #############################################################################
  #                             END OF TEST                                   #
  #############################################################################
  
#x = np.random.randn(10, 10)
#dout = np.random.randn(*x.shape)
#
#dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)
#
#_, cache = relu_forward(x)
#dx = relu_backward(dout, cache)
#
## The error should be around 1e-12
#print ('Testing relu_backward function:')
#print ('dx error: ', rel_error(dx_num, dx))
  
    #############################################################################
  #                             END OF TEST                                   #
  #############################################################################
# =============================================================================
# num_classes, num_inputs = 10, 50
# x = 0.001 * np.random.randn(num_inputs, num_classes)
# y = np.random.randint(num_classes, size=num_inputs)
# 
# dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
# loss, dx = svm_loss(x, y)
# 
# # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
# print ('Testing svm_loss:')
# print ('loss: ', loss)
# print ('dx error: ', rel_error(dx_num, dx))
# 
# dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
# loss, dx = softmax_loss(x, y)
# 
# # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
# print ('\nTesting softmax_loss:')
# print ('loss: ', loss)
# print ('dx error: ', rel_error(dx_num, dx))
# =============================================================================
  
# =============================================================================
# from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
# 
# x = np.random.randn(2, 3, 4)
# w = np.random.randn(12, 10)
# b = np.random.randn(10)
# dout = np.random.randn(2, 10)
# 
# out, cache = affine_relu_forward(x, w, b)
# dx, dw, db = affine_relu_backward(dout, cache)
# 
# dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)
# 
# print ('Testing affine_relu_forward:')
# print ('dx error: ', rel_error(dx_num, dx))
# print ('dw error: ', rel_error(dw_num, dw))
# print ('db error: ', rel_error(db_num, db))
# =============================================================================
  
# =============================================================================
# num_classes, num_inputs = 10, 50
# x = 0.001 * np.random.randn(num_inputs, num_classes)
# y = np.random.randint(num_classes, size=num_inputs)
# 
# dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
# loss, dx = svm_loss(x, y)
# 
# # Test svm_loss function. Loss should be around 9 and dx error should be 1e-9
# print ('Testing svm_loss:')
# print ('loss: ', loss)
# print ('dx error: ', rel_error(dx_num, dx))
# 
# dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
# loss, dx = softmax_loss(x, y)
# 
# # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
# print ('\nTesting softmax_loss:')
# print ('loss: ', loss)
# print ('dx error: ', rel_error(dx_num, dx))
# =============================================================================
# =============================================================================
#   
# N, D, H, C = 3, 5, 50, 7
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=N)
# 
# std = 1e-2
# model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
# 
# print ('Testing initialization ... ')
# W1_std = abs(model.params['W1'].std() - std)
# b1 = model.params['b1']
# W2_std = abs(model.params['W2'].std() - std)
# b2 = model.params['b2']
# assert W1_std < std / 10, 'First layer weights do not seem right'
# assert np.all(b1 == 0), 'First layer biases do not seem right'
# assert W2_std < std / 10, 'Second layer weights do not seem right'
# assert np.all(b2 == 0), 'Second layer biases do not seem right'
# 
# print ('Testing test-time forward pass ... ')
# model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
# model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
# model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
# model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
# X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
# scores = model.loss(X)
# correct_scores = np.asarray(
#   [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
#    [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
#    [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
# scores_diff = np.abs(scores - correct_scores).sum()
# assert scores_diff < 1e-6, 'Problem with test-time forward pass'
# 
# print ('Testing training loss (no regularization)')
# y = np.asarray([0, 5, 1])
# loss, grads = model.loss(X, y)
# correct_loss = 3.4702243556
# assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'
# 
# model.reg = 1.0
# loss, grads = model.loss(X, y)
# correct_loss = 26.5948426952
# assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'
# 
# for reg in [0.0, 0.7]:
#   print ('Running numeric gradient check with reg = ', reg)
#   model.reg = reg
#   loss, grads = model.loss(X, y)
# 
#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
#     print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#   
# =============================================================================
# =============================================================================
#   
# model = TwoLayerNet()
# solver = None
# 
# ##############################################################################
# # TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
# # 50% accuracy on the validation set.                                        #
# ##############################################################################
# data = get_CIFAR10_data()
# 
# #model = MyAwesomeModel(hidden_size=100, reg=10)
# solver = Solver(model, data,
#                   update_rule='sgd',
#                   optim_config={
#                     'learning_rate': 1e-3,
#                   },
#                   lr_decay=0.95,
#                   num_epochs=10, 
#                   batch_size=100,
#                   print_every=100)
# solver.train()
# ##############################################################################
# #                             END OF YOUR CODE                               #
# ##############################################################################  
#   
#   # Run this cell to visualize training loss and train / val accuracy
# 
# plt.subplot(2, 1, 1)
# plt.title('Training loss')
# plt.plot(solver.loss_history, 'o')
# plt.xlabel('Iteration')
# 
# plt.subplot(2, 1, 2)
# plt.title('Accuracy')
# plt.plot(solver.train_acc_history, '-o', label='train')
# plt.plot(solver.val_acc_history, '-o', label='val')
# plt.plot([0.5] * len(solver.val_acc_history), 'k--')
# plt.xlabel('Epoch')
# plt.legend(loc='lower right')
# plt.gcf().set_size_inches(15, 12)
# plt.show()
# =============================================================================
# =============================================================================
# 
# N, D, H1, H2, C = 2, 15, 20, 30, 10
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=(N,))
# 
# for reg in [0, 3.14]:
#   print ('Running check with reg = ', reg)
#   model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
#                             reg=reg, weight_scale=5e-2, dtype=np.float64)
# 
#   loss, grads = model.loss(X, y)
#   print ('Initial loss: ', loss)
# 
#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#     print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
# =============================================================================
# =============================================================================
# #data = get_CIFAR10_data()
# num_train = 50
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }
# 
# weight_scale = 0
# learning_rate = 1e-3
# model = FullyConnectedNet([100, 100],
#               weight_scale=weight_scale, dtype=np.float64)
# solver = Solver(model, small_data,
#                 print_every=10, num_epochs=20, batch_size=25,
#                 update_rule='sgd',
#                 optim_config={
#                   'learning_rate': learning_rate,
#                 }
#          )
# solver.train()
# 
# plt.plot(solver.loss_history, 'o')
# plt.title('Training loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Training loss')
# plt.show()
# =============================================================================
#import math
# =============================================================================
# #data = get_CIFAR10_data()
# num_train = 50
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }
# 
# learning_rate = 1e-2
# weight_scale = 5e-2#5e-2#math.sqrt(50)
# model = FullyConnectedNet([100, 100, 100, 100],
#                 weight_scale=weight_scale, dtype=np.float64)
# solver = Solver(model, small_data,
#                 print_every=10, num_epochs=20, batch_size=25,
#                 update_rule='sgd',
#                 optim_config={
#                   'learning_rate': learning_rate,
#                 }
#          )
# solver.train()
# 
# plt.plot(solver.loss_history, 'o')
# plt.title('Training loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Training loss')
# plt.show()
# =============================================================================
# =============================================================================
# from cs231n.optim import sgd_momentum
# 
# N, D = 4, 5
# w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
# dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
# v = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
# 
# config = {'learning_rate': 1e-3, 'velocity': v}
# next_w, _ = sgd_momentum(w, dw, config=config)
# 
# expected_next_w = np.asarray([
#   [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
#   [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
#   [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
#   [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])
# expected_velocity = np.asarray([
#   [ 0.5406,      0.55475789,  0.56891579, 0.58307368,  0.59723158],
#   [ 0.61138947,  0.62554737,  0.63970526,  0.65386316,  0.66802105],
#   [ 0.68217895,  0.69633684,  0.71049474,  0.72465263,  0.73881053],
#   [ 0.75296842,  0.76712632,  0.78128421,  0.79544211,  0.8096    ]])
# 
# print ('next_w error: ', rel_error(next_w, expected_next_w))
# print ('velocity error: ', rel_error(expected_velocity, config['velocity']))
# =============================================================================
# =============================================================================
# from cs231n.optim import rmsprop
# 
# N, D = 4, 5
# w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
# dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
# cache = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
# 
# config = {'learning_rate': 1e-2, 'cache': cache}
# next_w, _ = rmsprop(w, dw, config=config)
# 
# expected_next_w = np.asarray([
#   [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
#   [-0.132737,   -0.08078555, -0.02881884,  0.02316247,  0.07515774],
#   [ 0.12716641,  0.17918792,  0.23122175,  0.28326742,  0.33532447],
#   [ 0.38739248,  0.43947102,  0.49155973,  0.54365823,  0.59576619]])
# expected_cache = np.asarray([
#   [ 0.5976,      0.6126277,   0.6277108,   0.64284931,  0.65804321],
#   [ 0.67329252,  0.68859723,  0.70395734,  0.71937285,  0.73484377],
#   [ 0.75037008,  0.7659518,   0.78158892,  0.79728144,  0.81302936],
#   [ 0.82883269,  0.84469141,  0.86060554,  0.87657507,  0.8926    ]])
# 
# print ('next_w error: ', rel_error(expected_next_w, next_w))
# print ('cache error: ', rel_error(expected_cache, config['cache']))
# =============================================================================
# =============================================================================
# 
# from cs231n.optim import adam
# 
# N, D = 4, 5
# w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
# dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)
# m = np.linspace(0.6, 0.9, num=N*D).reshape(N, D)
# v = np.linspace(0.7, 0.5, num=N*D).reshape(N, D)
# 
# config = {'learning_rate': 1e-2, 'm': m, 'v': v, 't': 5}
# next_w, _ = adam(w, dw, config=config)
# 
# expected_next_w = np.asarray([
#   [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
#   [-0.1380274,  -0.08544591, -0.03286534,  0.01971428,  0.0722929],
#   [ 0.1248705,   0.17744702,  0.23002243,  0.28259667,  0.33516969],
#   [ 0.38774145,  0.44031188,  0.49288093,  0.54544852,  0.59801459]])
# expected_v = np.asarray([
#   [ 0.69966,     0.68908382,  0.67851319,  0.66794809,  0.65738853,],
#   [ 0.64683452,  0.63628604,  0.6257431,   0.61520571,  0.60467385,],
#   [ 0.59414753,  0.58362676,  0.57311152,  0.56260183,  0.55209767,],
#   [ 0.54159906,  0.53110598,  0.52061845,  0.51013645,  0.49966,   ]])
# expected_m = np.asarray([
#   [ 0.48,        0.49947368,  0.51894737,  0.53842105,  0.55789474],
#   [ 0.57736842,  0.59684211,  0.61631579,  0.63578947,  0.65526316],
#   [ 0.67473684,  0.69421053,  0.71368421,  0.73315789,  0.75263158],
#   [ 0.77210526,  0.79157895,  0.81105263,  0.83052632,  0.85      ]])
# 
# print ('next_w error: ', rel_error(expected_next_w, next_w))
# print ('v error: ', rel_error(expected_v, config['v']))
# print ('m error: ', rel_error(expected_m, config['m']))
# =============================================================================
# =============================================================================
# #data = get_CIFAR10_data()
# #num_train = 50000
# #small_data = {
# #   'X_train': data['X_train'][:num_train],
# #   'y_train': data['y_train'][:num_train],
# #   'X_val': data['X_val'], 
# #   'y_val': data['y_val'],
# # }
# # 
# #learning_rate = 1e-2
# #weight_scale = 5e-2#5e-2#math.sqrt(50)
# #model = FullyConnectedNet([100, 100, 100, 100],
# #                 weight_scale=weight_scale, dtype=np.float64)
# #solver = Solver(model, small_data,
# #                 print_every=10, num_epochs=20, batch_size=25,
# #                 update_rule='sgd',
# #                 optim_config={
# #                   'learning_rate': learning_rate,
# #                 }
# #          )
# #solver.train()
# # 
# #plt.plot(solver.loss_history, 'o')
# #plt.title('Training loss history')
# #plt.xlabel('Iteration')
# #plt.ylabel('Training loss')
# #plt.show()
# best_model = model
# y_test_pred = np.argmax(best_model.loss(X_test), axis=1)
# y_val_pred = np.argmax(best_model.loss(X_val), axis=1)
# print ('Validation set accuracy: ', (y_val_pred == y_val).mean())
# print ('Test set accuracy: ', (y_test_pred == y_test).mean())
# =============================================================================
# =============================================================================
#   
# N, D1, D2, D3 = 200, 50, 60, 3
# X = np.random.randn(N, D1)
# W1 = np.random.randn(D1, D2)
# W2 = np.random.randn(D2, D3)
# a = np.maximum(0, X.dot(W1)).dot(W2)
# 
# print ('Before batch normalization:')
# print ('  means: ', a.mean(axis=0))
# print ('  stds: ', a.std(axis=0))
# 
# # Means should be close to zero and stds close to one
# print ('After batch normalization (gamma=1, beta=0)')
# a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
# print ('  mean: ', a_norm.mean(axis=0))
# print ('  std: ', a_norm.std(axis=0))
# 
# # Now means should be close to beta and stds close to gamma
# gamma = np.asarray([1.0, 2.0, 3.0])
# beta = np.asarray([11.0, 12.0, 13.0])
# a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
# print ('After batch normalization (nontrivial gamma, beta)')
# print ('  means: ', a_norm.mean(axis=0))
# print ('  stds: ', a_norm.std(axis=0))
# 
# =============================================================================
# =============================================================================
# # Check the test-time forward pass by running the training-time
# # forward pass many times to warm up the running averages, and then
# # checking the means and variances of activations after a test-time
# # forward pass.
# 
# N, D1, D2, D3 = 200, 50, 60, 3
# W1 = np.random.randn(D1, D2)
# W2 = np.random.randn(D2, D3)
# 
# bn_param = {'mode': 'train'}
# gamma = np.ones(D3)
# beta = np.zeros(D3)
# for t in range(50):
#   X = np.random.randn(N, D1)
#   a = np.maximum(0, X.dot(W1)).dot(W2)
#   batchnorm_forward(a, gamma, beta, bn_param)
# bn_param['mode'] = 'test'
# X = np.random.randn(N, D1)
# a = np.maximum(0, X.dot(W1)).dot(W2)
# a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)
# 
# # Means should be close to zero and stds close to one, but will be
# # noisier than training-time forward passes.
# print ('After batch normalization (test-time):')
# print ('  means: ', a_norm.mean(axis=0))
# print ('  stds: ', a_norm.std(axis=0))
# =============================================================================
# Gradient check batchnorm backward pass
# =============================================================================
# 
# N, D = 4, 5
# x = 5 * np.random.randn(N, D) + 12
# gamma = np.random.randn(D)
# beta = np.random.randn(D)
# dout = np.random.randn(N, D)
# 
# bn_param = {'mode': 'train'}
# fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
# fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]
# fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]
# 
# dx_num = eval_numerical_gradient_array(fx, x, dout)
# da_num = eval_numerical_gradient_array(fg, gamma, dout)
# db_num = eval_numerical_gradient_array(fb, beta, dout)
# 
# _, cache = batchnorm_forward(x, gamma, beta, bn_param)
# dx, dgamma, dbeta = batchnorm_backward(dout, cache)
# print ('dx error: ', rel_error(dx_num, dx))
# print ('dgamma error: ', rel_error(da_num, dgamma))
# print ('dbeta error: ', rel_error(db_num, dbeta))
# =============================================================================
# =============================================================================
# N, D = 100, 500
# x = 5 * np.random.randn(N, D) + 12
# gamma = np.random.randn(D)
# beta = np.random.randn(D)
# dout = np.random.randn(N, D)
# 
# bn_param = {'mode': 'train'}
# out, cache = batchnorm_forward(x, gamma, beta, bn_param)
# 
# t1 = time.time()
# dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)
# t2 = time.time()
# dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)
# t3 = time.time()
# 
# print ('dx difference: ', rel_error(dx1, dx2))
# print ('dgamma difference: ', rel_error(dgamma1, dgamma2))
# print ('dbeta difference: ', rel_error(dbeta1, dbeta2))
# print ('speedup: %.2fx' % ((t2 - t1) / (t3 - t2)))
# =============================================================================
# =============================================================================
# N, D, H1, H2, C = 2, 15, 20, 30, 10
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=(N,))
# 
# for reg in [0, 3.14]:
#   print ('Running check with reg = ', reg)
#   model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
#                             reg=reg, weight_scale=5e-2, dtype=np.float64,
#                             use_batchnorm=True)
# 
#   loss, grads = model.loss(X, y)
#   print ('Initial loss: ', loss)
# 
#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#     print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#   if reg == 0: print
# 
# =============================================================================
# =============================================================================
#  Try training a very deep net with batchnorm
#data = get_CIFAR10_data()
#hidden_dims = [100, 100, 100, 100, 100]
# 
#num_train = 1000
#small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }
# 
#weight_scale = 2e-2
#bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
#model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)
# 
#bn_solver = Solver(bn_model, small_data,
#                 num_epochs=10, batch_size=50,
#                 update_rule='adam',
#                 optim_config={
#                   'learning_rate': 1e-3,
#                 },
#                 verbose=True, print_every=200)
#bn_solver.train()
# 
#solver = Solver(model, small_data,
#                 num_epochs=10, batch_size=50,
#                 update_rule='adam',
#                 optim_config={
#                   'learning_rate': 1e-3,
#                 },
#                 verbose=True, print_every=200)
#solver.train()
#   
#plt.subplot(3, 1, 1)
#plt.title('Training loss')
#plt.xlabel('Iteration')
# 
#plt.subplot(3, 1, 2)
#plt.title('Training accuracy')
#plt.xlabel('Epoch')
# 
#plt.subplot(3, 1, 3)
#plt.title('Validation accuracy')
#plt.xlabel('Epoch')
# 
#plt.subplot(3, 1, 1)
#plt.plot(solver.loss_history, 'o', label='baseline')
#plt.plot(bn_solver.loss_history, 'o', label='batchnorm')
# 
#plt.subplot(3, 1, 2)
#plt.plot(solver.train_acc_history, '-o', label='baseline')
#plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')
# 
#plt.subplot(3, 1, 3)
#plt.plot(solver.val_acc_history, '-o', label='baseline')
#plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')
#   
#for i in [1, 2, 3]:
#    
#    plt.subplot(3, 1, i)
#    plt.legend(loc='upper center', ncol=2)
#plt.gcf().set_size_inches(15, 15)
#plt.show()
# =============================================================================
# Try training a very deep net with batchnorm
# =============================================================================
# data = get_CIFAR10_data() 
# hidden_dims = [50, 50, 50, 50, 50, 50, 50]
# 
# num_train = 1000
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }
# 
# bn_solvers = {}
# solvers = {}
# weight_scales = np.logspace(-4, 0, num=20)
# for i, weight_scale in enumerate(weight_scales):
#   print ('Running weight scale %d / %d' % (i + 1, len(weight_scales)))
#   bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)
#   model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)
# 
#   bn_solver = Solver(bn_model, small_data,
#                   num_epochs=10, batch_size=50,
#                   update_rule='adam',
#                   optim_config={
#                     'learning_rate': 1e-3,
#                   },
#                   verbose=False, print_every=200)
#   bn_solver.train()
#   bn_solvers[weight_scale] = bn_solver
# 
#   solver = Solver(model, small_data,
#                   num_epochs=10, batch_size=50,
#                   update_rule='adam',
#                   optim_config={
#                     'learning_rate': 1e-3,
#                   },
#                   verbose=False, print_every=200)
#   solver.train()
#   solvers[weight_scale] = solver
# best_train_accs, bn_best_train_accs = [], []
# best_val_accs, bn_best_val_accs = [], []
# final_train_loss, bn_final_train_loss = [], []
# 
# for ws in weight_scales:
#   best_train_accs.append(max(solvers[ws].train_acc_history))
#   bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))
#   
#   best_val_accs.append(max(solvers[ws].val_acc_history))
#   bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))
#   
#   final_train_loss.append(np.mean(solvers[ws].loss_history[-100:]))
#   bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))
#   
# plt.subplot(3, 1, 1)
# plt.title('Best val accuracy vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Best val accuracy')
# plt.semilogx(weight_scales, best_val_accs, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')
# plt.legend(ncol=2, loc='lower right')
# 
# plt.subplot(3, 1, 2)
# plt.title('Best train accuracy vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Best training accuracy')
# plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')
# plt.legend()
# 
# plt.subplot(3, 1, 3)
# plt.title('Final training loss vs weight initialization scale')
# plt.xlabel('Weight initialization scale')
# plt.ylabel('Final training loss')
# plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')
# plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')
# plt.legend()
# 
# plt.gcf().set_size_inches(10, 15)
# plt.show()
# =============================================================================
# =============================================================================
# x = np.random.randn(500, 500) + 10
# 
# for p in [0.3, 0.6, 0.75]:
#   out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
#   out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})
# 
#   print ('Running tests with p = ', )
#   print ('Mean of input: ', x.mean())
#   print ('Mean of train-time output: ', out.mean())
#   print ('Mean of test-time output: ', out_test.mean())
#   print ('Fraction of train-time output set to zero: ', (out == 0).mean())
#   print ('Fraction of test-time output set to zero: ', (out_test == 0).mean())
#   print
# =============================================================================
# =============================================================================
# x = np.random.randn(10, 10) + 10
# dout = np.random.randn(*x.shape)
# 
# dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}
# out, cache = dropout_forward(x, dropout_param)
# dx = dropout_backward(dout, cache)
# dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)
# 
# print ('dx relative error: ', rel_error(dx, dx_num))
# =============================================================================
# =============================================================================
# N, D, H1, H2, C = 2, 15, 20, 30, 10
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=(N,))
# 
# for dropout in [0, 0.25, 0.5]:
#   print ('Running check with dropout = ', dropout)
#   model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
#                             weight_scale=5e-2, dtype=np.float64,
#                             dropout=dropout, seed=123)
# 
#   loss, grads = model.loss(X, y)
#   print ('Initial loss: ', loss)
# 
#   for name in sorted(grads):
#     f = lambda _: model.loss(X, y)[0]
#     grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
#     print ('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
#   print
# =============================================================================
# =============================================================================
# data = get_CIFAR10_data() 
# num_train = 500
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
# }
# 
# solvers = {}
# dropout_choices = [0, 0.75]
# for dropout in dropout_choices:
#   model = FullyConnectedNet([500], dropout=dropout)
#   print (dropout)
# 
#   solver = Solver(model, small_data,
#                   num_epochs=25, batch_size=100,
#                   update_rule='adam',
#                   optim_config={
#                     'learning_rate': 5e-4,
#                   },
#                   verbose=True, print_every=100)
#   solver.train()
#   solvers[dropout] = solver
# train_accs = []
# val_accs = []
# for dropout in dropout_choices:
#   solver = solvers[dropout]
#   train_accs.append(solver.train_acc_history[-1])
#   val_accs.append(solver.val_acc_history[-1])
# 
# plt.subplot(3, 1, 1)
# for dropout in dropout_choices:
#   plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
# plt.title('Train accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(ncol=2, loc='lower right')
#   
# plt.subplot(3, 1, 2)
# for dropout in dropout_choices:
#   plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
# plt.title('Val accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(ncol=2, loc='lower right')
# 
# plt.gcf().set_size_inches(15, 15)
# plt.show()
# =============================================================================
# =============================================================================
# x_shape = (2, 3, 4, 4)
# w_shape = (3, 3, 4, 4)
# x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
# w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
# b = np.linspace(-0.1, 0.2, num=3)
# 
# conv_param = {'stride': 2, 'pad': 1}
# out, _ = conv_forward_naive(x, w, b, conv_param)
# correct_out = np.array([[[[[-0.08759809, -0.10987781],
#                            [-0.18387192, -0.2109216 ]],
#                           [[ 0.21027089,  0.21661097],
#                            [ 0.22847626,  0.23004637]],
#                           [[ 0.50813986,  0.54309974],
#                            [ 0.64082444,  0.67101435]]],
#                          [[[-0.98053589, -1.03143541],
#                            [-1.19128892, -1.24695841]],
#                           [[ 0.69108355,  0.66880383],
#                            [ 0.59480972,  0.56776003]],
#                           [[ 2.36270298,  2.36904306],
#                            [ 2.38090835,  2.38247847]]]]])
# 
# # Compare your output to ours; difference should be around 1e-8
# print ('Testing conv_forward_naive')
# print ('difference: ', rel_error(out, correct_out))
# =============================================================================
# =============================================================================
# from scipy.misc import imread, imresize
# 
# kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
# # kitten is wide, and puppy is already square
# d = kitten.shape[1] - kitten.shape[0]
# kitten_cropped = kitten[:, int(d/2):int(-d/2), :]
# 
# img_size = 200   # Make this smaller if it runs too slow
# x = np.zeros((2, 3, img_size, img_size))
# x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
# x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))
# 
# # Set up a convolutional weights holding 2 filters, each 3x3
# w = np.zeros((2, 3, 3, 3))
# 
# # The first filter converts the image to grayscale.
# # Set up the red, green, and blue channels of the filter.
# w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
# w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
# w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
# 
# # Second filter detects horizontal edges in the blue channel.
# w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
# 
# # Vector of biases. We don't need any bias for the grayscale
# # filter, but for the edge detection filter we want to add 128
# # to each output so that nothing is negative.
# b = np.array([0, 128])
# 
# # Compute the result of convolving each input in x with each filter in w,
# # offsetting by b, and storing the results in out.
# out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})
# 
# def imshow_noax(img, normalize=True):
#     """ Tiny helper to show images as uint8 and remove axis labels """
#     if normalize:
#         img_max, img_min = np.max(img), np.min(img)
#         img = 255.0 * (img - img_min) / (img_max - img_min)
#     plt.imshow(img.astype('uint8'))
#     plt.gca().axis('off')
# 
# # Show the original images and the results of the conv operation
# plt.subplot(2, 3, 1)
# imshow_noax(puppy, normalize=False)
# plt.title('Original image')
# plt.subplot(2, 3, 2)
# imshow_noax(out[0, 0])
# plt.title('Grayscale')
# plt.subplot(2, 3, 3)
# imshow_noax(out[0, 1])
# plt.title('Edges')
# plt.subplot(2, 3, 4)
# imshow_noax(kitten_cropped, normalize=False)
# plt.subplot(2, 3, 5)
# imshow_noax(out[1, 0])
# plt.subplot(2, 3, 6)
# imshow_noax(out[1, 1])
# plt.show()
# =============================================================================
# =============================================================================
# x = np.random.randn(4, 3, 5, 5)
# w = np.random.randn(2, 3, 3, 3)
# b = np.random.randn(2,)
# dout = np.random.randn(4, 2, 5, 5)
# conv_param = {'stride': 1, 'pad': 1}
# 
# dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)
# 
# out, cache = conv_forward_naive(x, w, b, conv_param)
# dx, dw, db = conv_backward_naive(dout, cache)
# 
# # Your errors should be around 1e-9'
# print ('Testing conv_backward_naive function')
# print ('dx error: ', rel_error(dx, dx_num))
# print ('dw error: ', rel_error(dw, dw_num))
# print ('db error: ', rel_error(db, db_num))
# =============================================================================
x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = max_pool_forward_naive(x, pool_param)

correct_out = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])

# Compare your output with ours. Difference should be around 1e-8.
print ('Testing max_pool_forward_naive function:')
print ('difference: ', rel_error(out, correct_out))