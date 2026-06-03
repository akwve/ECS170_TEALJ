from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):
        self.method.data = self.dataset.load()
        result = self.method.run()
        self.result.data = result
        self.result.save()
        idx_test = self.method.data['train_test_val']['idx_test']
        self.evaluate.data = {
            "pred_y": result["pred_y"],
            "true_y": result["true_y"],
            "mask": idx_test
        }
        metrics = self.evaluate.evaluate()
        return metrics, None