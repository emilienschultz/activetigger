import { FC } from 'react';

export interface DisplayMatrixProps {
  labels: string[];
  matrix: number[][];
}

// component
export const DisplayMatrix: FC<DisplayMatrixProps> = ({ matrix, labels }) => {
  return (
    <div className="overflow-x-auto p-4">
      <table className="border-collapse border border-gray-300 w-full text-center">
        <thead>
          <tr className="bg-gray-200">
            <th className="border border-gray-300 p-2"></th>
            {labels.map((label, i) => (
              <th key={i} className="border border-gray-300 p-2">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-gray-100' : 'bg-white'}>
              <th className="border border-gray-300 p-2 bg-gray-200">{labels[i]}</th>
              {row.map((val, j) => (
                <td key={j} className="border border-gray-300 p-2">
                  {val}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
