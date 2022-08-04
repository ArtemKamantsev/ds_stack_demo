from unittest import TestCase, skip, skipIf, skipUnless, SkipTest, expectedFailure


class TestMethodSkipping(TestCase):
    @skip('demonstrating skipping')
    def test_skipping1(self) -> None:
        self.fail()

    # pylint: disable=comparison-of-constants
    @skipIf(0 < 1, 'demonstrating skipping')
    def test_skipping2(self) -> None:
        self.fail()

    # pylint: disable=comparison-of-constants
    @skipUnless(0 > 1, 'demonstrating skipping')
    def test_skipping3(self) -> None:
        self.fail()

    def test_skipping4(self) -> None:
        raise SkipTest('demonstrating skipping')

    def test_skipping5(self) -> None:
        self.skipTest('demonstrating skipping')
        self.fail()

    @expectedFailure
    def test_skipping6(self) -> None:
        with self.subTest(value=0):
            self.assertEqual(0, 0, 'correct')

        with self.subTest(value=1):
            self.assertEqual(0, 1, 'broken')


@skip('demonstrating skipping')
class TestClassSkipping(TestCase):
    def test(self) -> None:
        self.fail('demonstrating skipping')
