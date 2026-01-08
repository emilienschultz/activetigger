import { FC } from 'react';

export const MiddleEllipsis: FC<{ label: string }> = ({ label }) => {
  const splitIndex = Math.ceil(label.length / 2);
  return (
    <span className="truncate-middle">
      <span>{label.slice(0, splitIndex || 1)}</span>
      <span>{label.slice(splitIndex)}</span>
    </span>
  );
};
