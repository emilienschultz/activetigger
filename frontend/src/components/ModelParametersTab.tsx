import { FC } from 'react';

interface ModelParametersTabProps {
  params: Record<string, unknown>;
}

export const ModelParametersTab: FC<ModelParametersTabProps> = ({ params }) => {
  return (
    <table id="parameter-tables-thin">
      <thead>
        <tr>
          <th scope="col">Key</th>
          <th scope="col">Value</th>
        </tr>
      </thead>
      <tbody>
        {params &&
          Object.entries(params)
            .sort(([keyA], [keyB]) => {
              if (keyA === 'base_model') return -1;
              if (keyB === 'base_model') return 1;
              return 0;
            })
            .map(([key, value], index) => (
              <tr key={key} className={index % 2 === 0 ? 'dark' : ''}>
                <td>{key}</td>
                <td>{JSON.stringify(value)}</td>
              </tr>
            ))}
      </tbody>
    </table>
  );
};
