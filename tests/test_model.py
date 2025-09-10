from mesa_frames import Model


class CustomModel(Model):
    def __init__(self):
        super().__init__()
        self.custom_step_count = 0

    def step(self):
        self.custom_step_count += 2


class Test_Model:
    def test_steps(self):
        model = Model()

        assert model.steps == 0

        model.step()
        assert model.steps == 1

        model.step()
        assert model.steps == 2

    def test_user_defined_step(self):
        model = CustomModel()

        assert model.steps == 0
        assert model.custom_step_count == 0

        model.step()
        assert model.steps == 1
        assert model.custom_step_count == 2

        model.step()
        assert model.steps == 2
        assert model.custom_step_count == 4
