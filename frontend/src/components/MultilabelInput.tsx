import { FC, useState } from 'react';
import { FaSquareCheck } from 'react-icons/fa6';
import Select from 'react-select';

interface MulticlassInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string) => void;
}

export const MultilabelInput: FC<MulticlassInputProps> = ({
  elementId,
  postAnnotation,
  labels,
}) => {
  // management multilabels
  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);

  return (
    <div className="col-8 d-flex justify-content-center align-items-center">
      <Select
        isMulti
        className="flex-grow-1"
        options={labels.map((e) => ({ value: e, label: e }))}
        onChange={(e) => {
          setSelectedLabels(e.map((e) => e.value));
        }}
        value={selectedLabels.map((e) => ({ value: e, label: e }))}
      />
      <button
        className="btn"
        onClick={() => {
          postAnnotation(selectedLabels.join('|'), elementId);
          setSelectedLabels([]);
        }}
      >
        <FaSquareCheck size={30} />
      </button>
    </div>
  );
};
