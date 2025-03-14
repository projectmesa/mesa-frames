from mesa_frames import ModelDF


class Test_ModelDF:
    def test_steps(self):
        model = ModelDF()

        assert model.steps == 0

        model.step()
        assert model.steps == 1

        model.step()
        assert model.steps == 2
