import sys
import unittest

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reference"))
import bp_training_reference as reference


class BPComparisonPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected = reference.select_indices()
        cls.train_labels = reference.load_mnist()[1]

    def test_order_is_deterministic_and_seeded(self):
        first = reference.party_training_indices(self.selected, 123)
        second = reference.party_training_indices(self.selected, 123)
        changed = reference.party_training_indices(self.selected, 124)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_party_ownership_and_class_balance(self):
        parties = reference.party_training_indices(self.selected, 123)
        self.assertEqual([len(parties[p]) for p in range(3)], [30, 30, 30])
        for records in parties.values():
            counts = {label: 0 for label in range(10)}
            for index in records:
                counts[self.train_labels[index]] += 1
            self.assertEqual(set(counts.values()), {3})

    def test_split_sizes_and_unique_consumption(self):
        self.assertEqual([len(self.selected[name])
                          for name in ("train", "validation", "test")],
                         [90, 30, 30])
        for epoch in range(3):
            batches = reference.training_batches(self.selected, 123, epoch)
            consumed = [index for batch in batches for index in batch]
            self.assertEqual(len(batches), 30)
            self.assertEqual(len(consumed), 90)
            self.assertEqual(set(consumed), set(self.selected["train"]))
            self.assertTrue(all(len(set(batch)) == 3 for batch in batches))

    def test_new_schedule_is_not_old_class_major_schedule(self):
        old = reference.old_training_batches(self.selected)
        new = reference.training_batches(self.selected, 123)
        old_labels = [tuple(self.train_labels[index] for index in batch)
                      for batch in old]
        new_labels = [tuple(self.train_labels[index] for index in batch)
                      for batch in new]
        self.assertEqual(len(set(old_labels)), 10)
        self.assertGreater(len(set(new_labels)), 10)
        self.assertNotEqual(old, new)

    def test_initialization_is_reproducible_and_hashed(self):
        first = reference.initialize("he_xavier", 42)
        second = reference.initialize("he_xavier", 42)
        changed = reference.initialize("he_xavier", 43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first[0]), 784 * 8)
        self.assertEqual(len(first[2]), 8 * 10)
        self.assertNotEqual(reference.parameter_hash(first, "he_xavier", 42),
                            reference.parameter_hash(changed, "he_xavier", 43))


if __name__ == "__main__":
    unittest.main()
