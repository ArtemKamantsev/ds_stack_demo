from unittest import TestCase, skip, skipIf, skipUnless, SkipTest, expectedFailure

__all__ = ['TestMethodSkipping', 'TestClassSkipping']


class TestMethodSkipping(TestCase):
    @skip('demonstrating skipping')
    def test_skipping1(self):
        self.fail()

    @skipIf(0 < 1, 'demonstrating skipping')
    def test_skipping2(self):
        self.fail()

    @skipUnless(0 > 1, 'demonstrating skipping')
    def test_skipping3(self):
        self.fail()

    def test_skipping4(self):
        raise SkipTest('demonstrating skipping')

    def test_skipping5(self):
        self.skipTest('demonstrating skipping')
        self.fail()

    @expectedFailure
    def test_skipping6(self):
        with self.subTest(value=0):
            self.assertEqual(0, 0, 'correct')

        with self.subTest(value=1):
            self.assertEqual(0, 1, 'broken')


@skip('demonstrating skipping')
class TestClassSkipping(TestCase):
    def test(self):
        self.fail('demonstrating skipping')
