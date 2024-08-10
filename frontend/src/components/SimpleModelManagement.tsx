import { FC, useState } from 'react';
import { SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { FaRegTrashAlt } from 'react-icons/fa';
import { FaPlusCircle } from 'react-icons/fa';
import { RiFindReplaceLine } from 'react-icons/ri';

import { useUpdateSimpleModel } from '../core/api';
import { SimpleModelModel } from '../types';

interface SimpleModelManagementProps {
  projectName: string;
  currentScheme: string;
  currentModel: string;
  availableSimpleModels: { [key: string]: any };
  availableFeatures: string[];
}

export const SimpleModelManagement: FC<SimpleModelManagementProps> = ({
  projectName,
  currentScheme,
  currentModel,
  availableSimpleModels,
  availableFeatures,
}) => {
  // form management
  const { register, control, handleSubmit } = useForm<SimpleModelModel>({
    defaultValues: {
      features: [],
      model: '',
      scheme: currentScheme,
    },
  });

  // hooks to update
  const { updateSimpleModel } = useUpdateSimpleModel(projectName, currentScheme);

  // action when form validated
  const onSubmit: SubmitHandler<SimpleModelModel> = async (formData) => {
    await updateSimpleModel(formData);
  };

  return (
    <div className="d-flex align-items-center">
      <form onSubmit={handleSubmit(onSubmit)}>
        <label htmlFor="model">Select a model</label>
        <select className="form-control" id="model" {...register('model')}>
          {Object.values(availableSimpleModels).map((e) => (
            <option key={e}>{e}</option>
          ))}{' '}
        </select>
      </form>
    </div>
  );
};
