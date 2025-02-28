import { toPairs } from 'lodash';
import { FC } from 'react';

// Node caption
export const Caption: FC<{ labelColorMapping: { [key: string]: string } }> = ({
  labelColorMapping,
}) => {
  return (
    <div className="p-1">
      {toPairs(labelColorMapping).map(([label, color]) => (
        <div className="d-flex align-items-center" key={label}>
          <div className="node me-1" style={{ backgroundColor: color }} /> <span>{label}</span>
        </div>
      ))}
    </div>
  );
};
