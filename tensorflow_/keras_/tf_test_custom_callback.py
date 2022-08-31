from unittest import TestCase

import tensorflow as tf


class TestCustomCallback(TestCase):
    def test_custom_callback(self) -> None:
        action_history: list[tuple[str, ...]] = []
        callback_model: tf.keras.Model | None = None

        class CustomCallback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_train_begin', *keys))

            def on_train_end(self, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_train_end', *keys))

                nonlocal callback_model
                callback_model = self.model

            def on_epoch_begin(self, epoch, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_epoch_begin', f'epoch:{epoch}', *keys))

            def on_epoch_end(self, epoch, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_epoch_end', f'epoch:{epoch}', *keys))

            def on_test_begin(self, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_test_begin', *keys))

            def on_test_end(self, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_test_end', *keys))

            def on_predict_begin(self, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_predict_begin', *keys))

            def on_predict_end(self, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_predict_end', *keys))

            def on_train_batch_begin(self, batch, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_train_batch_begin', f'batch:{batch}', *keys))

            def on_train_batch_end(self, batch, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_train_batch_end', f'batch:{batch}', *keys))

            def on_test_batch_begin(self, batch, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_test_batch_begin', f'batch:{batch}', *keys))

            def on_test_batch_end(self, batch, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_test_batch_end', f'batch:{batch}', *keys))

            def on_predict_batch_begin(self, batch, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_predict_batch_begin', f'batch:{batch}', *keys))

            def on_predict_batch_end(self, batch, logs=None):
                keys = list(logs.keys())
                action_history.append(('on_predict_batch_end', f'batch:{batch}', *keys))

        input_ = tf.keras.layers.Input((1,))
        model = tf.keras.Model(inputs=input_, outputs=input_)
        model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=tf.keras.losses.mean_squared_error,
                metrics=tf.keras.metrics.MeanAbsoluteError(),
        )

        data_x = [[0], [2]]
        data_y = [2, 0]

        model.fit(data_x, data_y, batch_size=1, epochs=2, verbose=0, callbacks=[CustomCallback()])
        model.evaluate(data_x, data_y, batch_size=2, verbose=0, callbacks=[CustomCallback()])
        model.predict(data_x, batch_size=1, verbose=0, callbacks=[CustomCallback()])

        self.assertIs(model, callback_model)  # callback has an access to model it's used in
        self.assertListEqual(action_history, [('on_train_begin',),
                                              ('on_epoch_begin', 'epoch:0'),
                                              ('on_train_batch_begin', 'batch:0'),
                                              ('on_train_batch_end', 'batch:0', 'loss', 'mean_absolute_error'),
                                              ('on_train_batch_begin', 'batch:1'),
                                              ('on_train_batch_end', 'batch:1', 'loss', 'mean_absolute_error'),
                                              ('on_epoch_end', 'epoch:0', 'loss', 'mean_absolute_error'),
                                              ('on_epoch_begin', 'epoch:1'),
                                              ('on_train_batch_begin', 'batch:0'),
                                              ('on_train_batch_end', 'batch:0', 'loss', 'mean_absolute_error'),
                                              ('on_train_batch_begin', 'batch:1'),
                                              ('on_train_batch_end', 'batch:1', 'loss', 'mean_absolute_error'),
                                              ('on_epoch_end', 'epoch:1', 'loss', 'mean_absolute_error'),
                                              ('on_train_end', 'loss', 'mean_absolute_error'),
                                              ('on_test_begin',),
                                              ('on_test_batch_begin', 'batch:0'),
                                              ('on_test_batch_end', 'batch:0', 'loss', 'mean_absolute_error'),
                                              ('on_test_end', 'loss', 'mean_absolute_error'),
                                              ('on_predict_begin',),
                                              ('on_predict_batch_begin', 'batch:0'),
                                              ('on_predict_batch_end', 'batch:0', 'outputs'),
                                              ('on_predict_batch_begin', 'batch:1'),
                                              ('on_predict_batch_end', 'batch:1', 'outputs'),
                                              ('on_predict_end',)])
